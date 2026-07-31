// Command summarize calculates reproducible statistics from benchmark output files.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

type summary struct {
	SchemaVersion int `json:"schema_version"`
	Benchmark     struct {
		Cells       int64    `json:"cells"`
		ResultFiles []string `json:"result_files"`
	} `json:"benchmark"`
	Throughput []float64 `json:"throughput_cells_per_second"`
	Statistics struct {
		MedianCellsPerSecond               float64 `json:"median_cells_per_second"`
		MeanCellsPerSecond                 float64 `json:"mean_cells_per_second"`
		MinimumCellsPerSecond              float64 `json:"minimum_cells_per_second"`
		MaximumCellsPerSecond              float64 `json:"maximum_cells_per_second"`
		PopulationStandardDeviationPercent float64 `json:"population_standard_deviation_percent"`
	} `json:"statistics"`
}

func main() {
	directory := flag.String("directory", ".", "submission directory containing run-*.out/benchmark.txt")
	output := flag.String("output", "statistics.json", "output JSON path")
	expectedRuns := flag.Int("expected-runs", 5, "required number of benchmark runs")
	flag.Parse()

	if err := run(*directory, *output, *expectedRuns); err != nil {
		fmt.Fprintln(os.Stderr, "summarize:", err)
		os.Exit(1)
	}
}

func run(directory, output string, expectedRuns int) error {
	if expectedRuns < 1 {
		return fmt.Errorf("expected-runs must be positive")
	}
	paths, err := filepath.Glob(filepath.Join(directory, "run-*.out", "benchmark.txt"))
	if err != nil {
		return fmt.Errorf("find benchmark results: %w", err)
	}
	sort.Strings(paths)
	if len(paths) != expectedRuns {
		return fmt.Errorf("found %d benchmark results; expected %d", len(paths), expectedRuns)
	}

	var result summary
	result.SchemaVersion = 1
	for _, path := range paths {
		cells, throughput, err := parseResult(path)
		if err != nil {
			return err
		}
		if result.Benchmark.Cells == 0 {
			result.Benchmark.Cells = cells
		} else if cells != result.Benchmark.Cells {
			return fmt.Errorf("cell count changed from %d to %d in %s", result.Benchmark.Cells, cells, path)
		}
		relative, err := filepath.Rel(directory, path)
		if err != nil {
			return fmt.Errorf("make result path relative: %w", err)
		}
		result.Benchmark.ResultFiles = append(result.Benchmark.ResultFiles, filepath.ToSlash(relative))
		result.Throughput = append(result.Throughput, throughput)
	}

	ordered := append([]float64(nil), result.Throughput...)
	sort.Float64s(ordered)
	mean := 0.0
	for _, value := range ordered {
		mean += value
	}
	mean /= float64(len(ordered))
	variance := 0.0
	for _, value := range ordered {
		difference := value - mean
		variance += difference * difference
	}
	variance /= float64(len(ordered))
	median := ordered[len(ordered)/2]
	if len(ordered)%2 == 0 {
		median = (ordered[len(ordered)/2-1] + ordered[len(ordered)/2]) / 2
	}
	result.Statistics.MedianCellsPerSecond = median
	result.Statistics.MeanCellsPerSecond = mean
	result.Statistics.MinimumCellsPerSecond = ordered[0]
	result.Statistics.MaximumCellsPerSecond = ordered[len(ordered)-1]
	result.Statistics.PopulationStandardDeviationPercent = math.Round(math.Sqrt(variance)/mean*100*1000) / 1000

	file, err := os.Create(output)
	if err != nil {
		return fmt.Errorf("create %s: %w", output, err)
	}
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(result); err != nil {
		file.Close()
		return fmt.Errorf("write %s: %w", output, err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close %s: %w", output, err)
	}
	fmt.Printf("wrote %s: median %.2f M cells/s across %d runs\n", output, median/1e6, len(ordered))
	return nil
}

func parseResult(path string) (int64, float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, 0, fmt.Errorf("open %s: %w", path, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0, 0, fmt.Errorf("invalid benchmark row in %s", path)
		}
		cellsFloat, err := strconv.ParseFloat(fields[0], 64)
		if err != nil || cellsFloat <= 0 || math.Trunc(cellsFloat) != cellsFloat {
			return 0, 0, fmt.Errorf("invalid cell count in %s", path)
		}
		throughput, err := strconv.ParseFloat(fields[1], 64)
		if err != nil || throughput <= 0 {
			return 0, 0, fmt.Errorf("invalid throughput in %s", path)
		}
		return int64(cellsFloat), throughput, nil
	}
	if err := scanner.Err(); err != nil {
		return 0, 0, fmt.Errorf("read %s: %w", path, err)
	}
	return 0, 0, fmt.Errorf("no benchmark row in %s", path)
}
