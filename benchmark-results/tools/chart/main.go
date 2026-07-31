// Command chart regenerates the benchmark comparison SVG from repository data.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"html"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const (
	cudaSnapshotPath = "benchmark-results/apple-m4-macbook-air-20260731/website-gpu-ranking.txt"
	estimatesPath    = "benchmark-results/apple-silicon-estimates.json"
	registryPath     = "benchmark-results/registry.json"
	defaultOutput    = "benchmark-results/apple-silicon-vs-cuda.svg"
	expectedCUDA     = 58
	expectedFormula  = "reference_throughput * min(compute_ratio, memory_bandwidth_gbps / reference_memory_bandwidth_gbps)"
)

type estimateCatalog struct {
	SchemaVersion int `json:"schema_version"`
	Model         struct {
		Name                         string   `json:"name"`
		ReferenceLabel               string   `json:"reference_label"`
		ReferenceThroughputMCells    float64  `json:"reference_throughput_mcells_per_second"`
		ReferenceMemoryBandwidthGBPS float64  `json:"reference_memory_bandwidth_gbps"`
		Formula                      string   `json:"formula"`
		Scope                        string   `json:"scope"`
		Assumptions                  []string `json:"assumptions"`
	} `json:"model"`
	References []struct {
		ID    string `json:"id"`
		Title string `json:"title"`
		URL   string `json:"url"`
	} `json:"references"`
	Estimates []struct {
		Label               string   `json:"label"`
		GPUCores            int      `json:"gpu_cores"`
		MemoryBandwidthGBPS float64  `json:"memory_bandwidth_gbps"`
		ComputeRatio        float64  `json:"compute_ratio"`
		Sources             []string `json:"sources"`
	} `json:"estimates"`
}

type registry struct {
	SchemaVersion int `json:"schema_version"`
	Measurements  []struct {
		Label                 string `json:"label"`
		Backend               string `json:"backend"`
		ResultPath            string `json:"result_path"`
		EvidencePath          string `json:"evidence_path"`
		ReplacesEstimateLabel string `json:"replaces_estimate_label"`
	} `json:"measurements"`
}

type statisticsFile struct {
	Statistics struct {
		MedianCellsPerSecond float64 `json:"median_cells_per_second"`
	} `json:"statistics"`
}

type entry struct {
	Label string
	Kind  string
	Value float64
}

func main() {
	repositoryRoot := flag.String("root", ".", "repository root")
	outputPath := flag.String("output", defaultOutput, "SVG output path, relative to the repository root")
	flag.Parse()

	if err := run(*repositoryRoot, *outputPath); err != nil {
		fmt.Fprintln(os.Stderr, "chart:", err)
		os.Exit(1)
	}
}

func run(repositoryRoot, outputPath string) error {
	catalog, err := loadJSONStrict[estimateCatalog](filepath.Join(repositoryRoot, estimatesPath))
	if err != nil {
		return err
	}
	measurements, err := loadJSONStrict[registry](filepath.Join(repositoryRoot, registryPath))
	if err != nil {
		return err
	}
	if catalog.SchemaVersion != 1 || measurements.SchemaVersion != 1 {
		return fmt.Errorf("unsupported data schema")
	}
	if catalog.Model.Name == "" || catalog.Model.ReferenceLabel == "" || catalog.Model.Formula == "" || catalog.Model.Scope == "" || len(catalog.Model.Assumptions) == 0 {
		return fmt.Errorf("projection model metadata is incomplete")
	}
	if catalog.Model.Formula != expectedFormula {
		return fmt.Errorf("projection formula does not match the implemented model")
	}
	if catalog.Model.ReferenceThroughputMCells <= 0 || catalog.Model.ReferenceMemoryBandwidthGBPS <= 0 {
		return fmt.Errorf("projection reference values must be positive")
	}

	entries, err := loadCUDA(filepath.Join(repositoryRoot, cudaSnapshotPath))
	if err != nil {
		return err
	}
	if len(entries) != expectedCUDA {
		return fmt.Errorf("CUDA snapshot has %d entries; expected %d", len(entries), expectedCUDA)
	}

	referenceMeasured := false
	replaced := make(map[string]bool)
	measurementLabels := make(map[string]bool)
	for _, measurement := range measurements.Measurements {
		if measurement.Label == "" || measurement.Backend == "" || measurement.ResultPath == "" || measurement.EvidencePath == "" {
			return fmt.Errorf("registry measurement has an empty required field")
		}
		if measurement.Backend != "Metal" {
			return fmt.Errorf("chart measurement %q uses unsupported backend %q", measurement.Label, measurement.Backend)
		}
		if measurementLabels[measurement.Label] {
			return fmt.Errorf("measurement labels must be unique: %q", measurement.Label)
		}
		measurementLabels[measurement.Label] = true
		stats, err := loadJSON[statisticsFile](filepath.Join(repositoryRoot, filepath.FromSlash(measurement.ResultPath)))
		if err != nil {
			return err
		}
		value := stats.Statistics.MedianCellsPerSecond / 1e6
		if value <= 0 {
			return fmt.Errorf("%s has a non-positive median", measurement.ResultPath)
		}
		entries = append(entries, entry{Label: measurement.Label, Kind: "measured", Value: value})
		if _, err := os.Stat(filepath.Join(repositoryRoot, filepath.FromSlash(measurement.EvidencePath))); err != nil {
			return fmt.Errorf("measurement evidence %s: %w", measurement.EvidencePath, err)
		}
		if measurement.ReplacesEstimateLabel != "" {
			replaced[measurement.ReplacesEstimateLabel] = true
		}
		if strings.Contains(measurement.Label, catalog.Model.ReferenceLabel) {
			referenceMeasured = true
			if math.Abs(value-catalog.Model.ReferenceThroughputMCells) > 1e-9 {
				return fmt.Errorf("projection anchor %.12g does not match measured median %.12g", catalog.Model.ReferenceThroughputMCells, value)
			}
		}
	}
	if !referenceMeasured {
		return fmt.Errorf("registry does not contain projection reference %q", catalog.Model.ReferenceLabel)
	}

	referenceIDs := make(map[string]bool)
	for _, reference := range catalog.References {
		if reference.ID == "" || reference.Title == "" || !strings.HasPrefix(reference.URL, "https://") || referenceIDs[reference.ID] {
			return fmt.Errorf("references must have a unique ID, title, and HTTPS URL")
		}
		referenceIDs[reference.ID] = true
	}
	labels := make(map[string]bool)
	for _, estimate := range catalog.Estimates {
		if replaced[estimate.Label] {
			continue
		}
		if estimate.Label == "" || labels[estimate.Label] {
			return fmt.Errorf("estimate labels must be non-empty and unique")
		}
		labels[estimate.Label] = true
		if estimate.GPUCores <= 0 || estimate.ComputeRatio <= 0 || estimate.MemoryBandwidthGBPS <= 0 || len(estimate.Sources) == 0 {
			return fmt.Errorf("%s has incomplete projection inputs", estimate.Label)
		}
		for _, source := range estimate.Sources {
			if !referenceIDs[source] {
				return fmt.Errorf("%s cites unknown reference %q", estimate.Label, source)
			}
		}
		bandwidthRatio := estimate.MemoryBandwidthGBPS / catalog.Model.ReferenceMemoryBandwidthGBPS
		value := catalog.Model.ReferenceThroughputMCells * math.Min(estimate.ComputeRatio, bandwidthRatio)
		entries = append(entries, entry{Label: estimate.Label, Kind: "estimated", Value: value})
	}

	sort.SliceStable(entries, func(i, j int) bool {
		if entries[i].Value != entries[j].Value {
			return entries[i].Value > entries[j].Value
		}
		if entries[i].Kind != entries[j].Kind {
			return kindOrder(entries[i].Kind) < kindOrder(entries[j].Kind)
		}
		return entries[i].Label < entries[j].Label
	})

	svg, err := render(entries)
	if err != nil {
		return err
	}
	output := filepath.Join(repositoryRoot, filepath.FromSlash(outputPath))
	if err := os.WriteFile(output, svg, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", output, err)
	}
	fmt.Printf("wrote %s with %d entries (%d CUDA, %d measured, %d estimated)\n",
		output, len(entries), countKind(entries, "cuda"), countKind(entries, "measured"), countKind(entries, "estimated"))
	return nil
}

func loadJSON[T any](path string) (T, error) {
	var value T
	data, err := os.ReadFile(path)
	if err != nil {
		return value, fmt.Errorf("read %s: %w", path, err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	if err := decoder.Decode(&value); err != nil {
		return value, fmt.Errorf("decode %s: %w", path, err)
	}
	return value, nil
}

func loadJSONStrict[T any](path string) (T, error) {
	var value T
	data, err := os.ReadFile(path)
	if err != nil {
		return value, fmt.Errorf("read %s: %w", path, err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return value, fmt.Errorf("decode %s: %w", path, err)
	}
	return value, nil
}

func loadCUDA(path string) ([]entry, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer file.Close()

	var entries []entry
	labels := make(map[string]bool)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		firstQuote := strings.IndexByte(line, '"')
		lastQuote := strings.LastIndexByte(line, '"')
		if firstQuote < 0 || lastQuote <= firstQuote {
			return nil, fmt.Errorf("invalid CUDA snapshot line %q", line)
		}
		fields := strings.Fields(line[:firstQuote])
		if len(fields) < 2 {
			return nil, fmt.Errorf("invalid CUDA snapshot values in %q", line)
		}
		cellsPerSecond, err := strconv.ParseFloat(fields[1], 64)
		if err != nil || cellsPerSecond <= 0 {
			return nil, fmt.Errorf("invalid CUDA throughput in %q", line)
		}
		label := line[firstQuote+1 : lastQuote]
		if label == "" || labels[label] {
			return nil, fmt.Errorf("CUDA labels must be non-empty and unique: %q", label)
		}
		labels[label] = true
		entries = append(entries, entry{Label: label, Kind: "cuda", Value: cellsPerSecond / 1e6})
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	return entries, nil
}

func render(entries []entry) ([]byte, error) {
	if len(entries) == 0 {
		return nil, fmt.Errorf("cannot render an empty chart")
	}
	const (
		width      = 1500
		chartX     = 500
		chartWidth = 840
		rowHeight  = 25
		rowsY      = 198
		bottom     = 76
	)
	height := rowsY + len(entries)*rowHeight + bottom
	maxValue := entries[0].Value
	axisMaximum := math.Ceil(maxValue/250) * 250
	if axisMaximum < 250 {
		axisMaximum = 250
	}

	var output bytes.Buffer
	fmt.Fprintf(&output, `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" role="img" aria-labelledby="title description">
  <title id="title">mumax³ 4-million-cell benchmark comparison</title>
  <desc id="description">A ranked horizontal bar chart containing %d measured Apple Metal result(s), Apple Silicon projections, and 58 CUDA results published by the mumax³ website. Higher throughput is faster. Projections are not measurements.</desc>
  <style>
    text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #182230; }
    .title { font-size: 26px; font-weight: 700; }
    .subtitle { font-size: 14px; fill: #52606d; }
    .axis { font-size: 12px; fill: #697586; }
    .label { font-size: 13px; }
    .value { font-size: 12px; font-variant-numeric: tabular-nums; }
    .note { font-size: 12px; fill: #52606d; }
  </style>
  <rect width="100%%" height="100%%" fill="#ffffff"/>
  <text class="title" x="20" y="38">mumax³ 4M-cell benchmark: Apple Silicon and published CUDA results</text>
  <text class="subtitle" x="20" y="64">Combined ordering by throughput (higher is faster) · CUDA snapshot checked 2026-07-31</text>
  <g aria-label="Legend">
    <rect x="20" y="86" width="18" height="12" rx="2" fill="#1473e6"/>
    <text class="subtitle" x="46" y="97">Apple Metal — measured</text>
    <rect x="226" y="86" width="18" height="12" rx="2" fill="#e67e22"/>
    <text class="subtitle" x="252" y="97">Apple Metal — estimated</text>
    <rect x="446" y="86" width="18" height="12" rx="2" fill="#9aa0a6"/>
    <text class="subtitle" x="472" y="97">Published mumax³ CUDA result</text>
  </g>
  <text class="subtitle" x="20" y="128">Orange entries are chip-level projections anchored to the blue M4 measurement; they are not hardware measurements or official mumax³ ranks.</text>
  <text class="axis" x="20" y="176">Combined position and device</text>
  <text class="axis" x="%d" y="154" text-anchor="middle">Throughput (million cells/s)</text>
`, width, height, width, height, countKind(entries, "measured"), chartX+chartWidth/2)

	for tick := 0.0; tick <= axisMaximum+0.1; tick += 250 {
		x := chartX + int(tick/axisMaximum*chartWidth)
		fmt.Fprintf(&output, "  <line x1=\"%d\" y1=\"184\" x2=\"%d\" y2=\"%d\" stroke=\"#e4e7ec\" stroke-width=\"1\"/>\n", x, x, rowsY+len(entries)*rowHeight)
		fmt.Fprintf(&output, "  <text class=\"axis\" x=\"%d\" y=\"176\" text-anchor=\"middle\">%.0f</text>\n", x, tick)
	}

	for index, item := range entries {
		y := rowsY + index*rowHeight
		if index%2 == 1 {
			fmt.Fprintf(&output, "  <rect x=\"12\" y=\"%d\" width=\"1476\" height=\"%d\" fill=\"#f8fafc\"/>\n", y, rowHeight)
		}
		barWidth := int(item.Value / axisMaximum * chartWidth)
		if barWidth < 2 {
			barWidth = 2
		}
		barY := y + 6
		class, color, suffix := style(item.Kind)
		label := fmt.Sprintf("%d. %s%s", index+1, item.Label, suffix)
		tooltip := fmt.Sprintf("%s: %.2f million cells/s", label, item.Value)
		fmt.Fprintf(&output, "  <g class=\"entry %s\">\n    <title>%s</title>\n", class, escape(tooltip))
		fmt.Fprintf(&output, "    <text class=\"label\" x=\"20\" y=\"%d\">%s</text>\n", y+17, escape(label))
		fmt.Fprintf(&output, "    <rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"13\" rx=\"2\" fill=\"%s\"/>\n", chartX, barY, barWidth, color)
		fmt.Fprintf(&output, "    <text class=\"value\" x=\"%d\" y=\"%d\">%.2f</text>\n  </g>\n", chartX+barWidth+8, y+17, item.Value)
	}

	fmt.Fprintf(&output, `  <line x1="12" y1="%d" x2="1488" y2="%d" stroke="#d0d5dd"/>
  <text class="note" x="20" y="%d">Source: mumax³ website gpus.svg snapshot (58 CUDA entries) · Apple values and reproducible inputs: benchmark-results/</text>
  <text class="note" x="20" y="%d">Ranking in this chart is a comparison aid. Only gray and blue bars are measured; orange bars must be replaced by submitted measurements.</text>
</svg>
`, rowsY+len(entries)*rowHeight, rowsY+len(entries)*rowHeight, height-38, height-18)
	return output.Bytes(), nil
}

func style(kind string) (class, color, suffix string) {
	switch kind {
	case "cuda":
		return "published-cuda", "#9aa0a6", ""
	case "measured":
		return "measured-metal", "#1473e6", ""
	case "estimated":
		return "estimated-metal", "#e67e22", " (estimated)"
	default:
		panic("unknown entry kind: " + kind)
	}
}

func kindOrder(kind string) int {
	switch kind {
	case "measured":
		return 0
	case "cuda":
		return 1
	case "estimated":
		return 2
	default:
		return 3
	}
}

func countKind(entries []entry, kind string) int {
	count := 0
	for _, item := range entries {
		if item.Kind == kind {
			count++
		}
	}
	return count
}

func escape(value string) string {
	return html.EscapeString(value)
}
