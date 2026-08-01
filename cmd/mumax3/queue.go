package main

// File queue for distributing multiple input files over GPUs.

import (
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/engine"
	"github.com/mumax/3/util"
)

var (
	exitStatus       atom = 0
	numOK, numFailed atom = 0, 0
	// A single simulation cannot keep an Apple GPU busy once its step rate is
	// set by host round trips rather than by arithmetic. Measured on an M4 at
	// 128x128: three concurrent jobs finish in the wall-clock time of one, and
	// the aggregate rate stops improving past three or four. See -j.
	flag_jobs = flag.Int("j", 1, "Number of simulations to run concurrently on each GPU")
)

// A queue slot: which GPU to use, and a stable index used to give each
// concurrently running job its own GUI port.
type slot struct {
	gpu   int
	index int
}

func RunQueue(files []string) {
	s := NewStateTab(files)
	s.PrintTo(os.Stdout)
	go s.ListenAndServe(*engine.Flag_port)
	fmt.Print("//Realtime queue overview available at http://127.0.0.1", *engine.Flag_port, "\n")
	s.Run()
	fmt.Println(numOK.get(), "OK, ", numFailed.get(), "failed")
	os.Exit(int(exitStatus))
}

// StateTab holds the queue state (list of jobs + statuses).
// All operations are atomic.
type stateTab struct {
	lock sync.Mutex
	jobs []job
	next int
}

// Job info.
type job struct {
	inFile  string // input file to run
	webAddr string // http address for gui of running process
	uid     int
}

// NewStateTab constructs a queue for the given input files.
// After construction, it is accessed atomically.
func NewStateTab(inFiles []string) *stateTab {
	s := new(stateTab)
	s.jobs = make([]job, len(inFiles))
	for i, f := range inFiles {
		s.jobs[i] = job{inFile: f, uid: i}
	}
	return s
}

// StartNext advances the next job and marks it running, setting its webAddr to indicate the GUI url.
// A copy of the job info is returned, the original remains unmodified.
// ok is false if there is no next job.
func (s *stateTab) StartNext(webAddr string) (next job, ok bool) {
	s.lock.Lock()
	defer s.lock.Unlock()
	if s.next >= len(s.jobs) {
		return job{}, false
	}
	s.jobs[s.next].webAddr = webAddr
	jobCopy := s.jobs[s.next]
	s.next++
	return jobCopy, true
}

// Finish marks the job with j's uid as finished.
func (s *stateTab) Finish(j job) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.jobs[j.uid].webAddr = ""
}

// Runs all the jobs in stateTab.
func (s *stateTab) Run() {
	idle, nSlots := initSlots()
	for {
		sl := <-idle
		addr := ""
		if *engine.Flag_port != "" {
			_, p, _ := net.SplitHostPort(*engine.Flag_port)
			// +1 because Flag_port hosts the overview of the queue itself.
			addr = fmt.Sprint(":", atoi(p)+1+sl.index)
		}
		j, ok := s.StartNext(addr)
		if !ok {
			break
		}
		go func() {
			run(j.inFile, sl.gpu, j.webAddr)
			s.Finish(j)
			idle <- sl
		}()
	}
	// drain remaining tasks (one already done)
	for i := 1; i < nSlots; i++ {
		<-idle
	}
}

func atoi(a string) int {
	i, err := strconv.Atoi(a)
	util.PanicErr(err)
	return i
}

type atom int32

func (a *atom) set(v int) { atomic.StoreInt32((*int32)(a), int32(v)) }
func (a *atom) get() int  { return int(atomic.LoadInt32((*int32)(a))) }
func (a *atom) inc()      { atomic.AddInt32((*int32)(a), 1) }

func run(inFile string, gpu int, webAddr string) {
	// overridden flags
	gpuFlag := fmt.Sprint(`-gpu=`, gpu)
	httpFlag := fmt.Sprint(`-http=`, webAddr)

	// pass through flags
	flags := []string{gpuFlag, httpFlag}
	flag.Visit(func(f *flag.Flag) {
		// -j governs this queue, not the single-file children it spawns.
		if f.Name != "gpu" && f.Name != "http" && f.Name != "failfast" &&
			f.Name != "j" {
			flags = append(flags, fmt.Sprintf("-%v=%v", f.Name, f.Value))
		}
	})
	flags = append(flags, inFile)

	cmd := exec.Command(os.Args[0], flags...)
	log.Println(os.Args[0], flags)
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Println(inFile, err)
		log.Printf("%s\n", output)
		exitStatus.set(1)
		numFailed.inc()
		if *flag_failfast {
			os.Exit(1)
		}
	} else {
		numOK.inc()
	}
}

// Creates a concurrent channel of runnable slots. Each GPU contributes -j
// slots, so a queue can keep more than one simulation in flight per device.
// Returns the channel and the total number of slots.
func initSlots() (chan slot, int) {
	nGpu := cu.DeviceGetCount()
	if nGpu == 0 {
		log.Fatal("no GPUs available")
	}

	singleGPU := engine.FlagPassed("gpu")
	gpus := make([]int, 0, nGpu)
	if singleGPU {
		gpus = append(gpus, *engine.Flag_gpu)
	} else {
		for i := 0; i < nGpu; i++ {
			gpus = append(gpus, i)
		}
	}

	perGPU := *flag_jobs
	if perGPU < 1 {
		perGPU = 1
	}
	nSlots := len(gpus) * perGPU
	idle := make(chan slot, nSlots)
	index := 0
	for round := 0; round < perGPU; round++ {
		for _, gpu := range gpus {
			idle <- slot{gpu: gpu, index: index}
			index++
		}
	}
	if perGPU > 1 {
		log.Printf("//running %d simulations concurrently on each of %d GPU(s)",
			perGPU, len(gpus))
	}
	return idle, nSlots
}

func (s *stateTab) PrintTo(w io.Writer) {
	s.lock.Lock()
	defer s.lock.Unlock()
	for i, j := range s.jobs {
		fmt.Fprintf(w, "%3d %v %v\n", i, j.inFile, j.webAddr)
	}
}

func (s *stateTab) RenderHTML(w io.Writer) {
	s.lock.Lock()
	defer s.lock.Unlock()
	fmt.Fprintln(w, ` 
<!DOCTYPE html> <html> <head> 
	<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
	<meta http-equiv="refresh" content="1">
`+engine.CSS+`
	</head><body>
	<span style="color:gray; font-weight:bold; font-size:1.5em"> mumax<sup>3</sup> queue status </span><br/>
	<hr/>
	<pre>
`)

	hostname := "localhost"
	hostname, _ = os.Hostname()
	for _, j := range s.jobs {
		if j.webAddr != "" {
			fmt.Fprint(w, `<b>`, j.uid, ` <a href="`, "http://", hostname+j.webAddr, `">`, j.inFile, " ", j.webAddr, "</a></b>\n")
		} else {
			fmt.Fprint(w, j.uid, " ", j.inFile, "\n")
		}
	}

	fmt.Fprintln(w, `</pre><hr/></body></html>`)
}

func (s *stateTab) ListenAndServe(addr string) {
	http.Handle("/", s)
	go http.ListenAndServe(addr, nil)
}

func (s *stateTab) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.RenderHTML(w)
}
