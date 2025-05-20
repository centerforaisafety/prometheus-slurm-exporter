package main

import (
	"fmt"
	"io/ioutil"
	"os/exec"
	"strconv"
	"strings"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/common/log" // Using the same log as in gpus.go
)

// executeSlurmCommand is a helper function to run Slurm commands.
// For consistency with existing code, it uses log.Fatal on error.
// In a production system, returning an error might be preferable.
func executeSlurmCommand(command string, arguments []string) []byte {
	cmd := exec.Command(command, arguments...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatalf("Failed to create stdout pipe for command %s %v: %v", command, arguments, err)
	}
	if err := cmd.Start(); err != nil {
		log.Fatalf("Failed to start command %s %v: %v", command, arguments, err)
	}
	out, readErr := ioutil.ReadAll(stdout)
	if readErr != nil {
		log.Warnf("Failed to read all stdout from command %s %v: %v", command, arguments, readErr)
		// Continue to Wait, as the command might have finished with an error code
	}
	if err := cmd.Wait(); err != nil {
		// Log non-fatal error if output was partially read but command failed.
		// If readErr was nil, this indicates a failure in the command itself.
		log.Warnf("Command %s %v finished with error: %v. Output: %s", command, arguments, err, string(out))
        // Depending on strictness, could return nil or empty here.
        // Given existing fatal errors, if a command fails to run properly, it might be better to get no data.
        // However, if we have partial data (e.g. from squeue), sometimes it's still useful.
        // For now, let's return output even if Wait() reports an error, as squeue might still output data on some errors.
	}
	return out
}

// PartitionGPUMetrics holds the GPU metrics for a single partition.
type PartitionGPUMetrics struct {
	allocated float64
	idle      float64
	total     float64
}

// parseTotalGPUsPerPartition fetches and parses the total number of GPUs available per partition.
// It does this by summing GPUs from nodes belonging to each partition.
func parseTotalGPUsPerPartition() map[string]float64 {
	totals := make(map[string]float64)
	// Command: sinfo -N -h -o "%N %P %G" (NodeName, PartitionName, GRES on node)
	args := []string{"-N", "-h", "-o", "%N %P %G"}
	output := string(executeSlurmCommand("sinfo", args))

	for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
		parts := strings.Fields(line)
		// Expecting at least 3 parts: NodeName, PartitionName(s), GRES
		if len(parts) < 3 {
			continue
		}

		partitionNamesStr := parts[1]
		gresStr := parts[2]

		if strings.Contains(gresStr, "(null)") || !strings.Contains(strings.ToLower(gresStr), "gpu") {
			continue // No GRES, GRES is null, or GRES does not contain "gpu"
		}

		nodeGpuCount := 0.0
		potentialGpuEntries := strings.Split(gresStr, ",") // GRES can be comma-separated, e.g., "gpu:2,mps:200"

		for _, entry := range potentialGpuEntries {
			if !strings.HasPrefix(strings.ToLower(entry), "gpu") &&
			!strings.HasPrefix(strings.ToLower(entry), "gres/gpu") {
				continue
			}

			// Remove everything after first '('  →  "gpu:A100:8"
			entryClean := strings.Split(entry, "(")[0]

			// Remove leading "gpu:" or "gres/gpu:"
			entryClean = strings.TrimPrefix(entryClean, "gpu:")
			entryClean = strings.TrimPrefix(entryClean, "gres/gpu:")

			// Count is now the field after the last ':'
			valParts := strings.Split(entryClean, ":")
			countStr  := valParts[len(valParts)-1]

			if n, err := strconv.ParseFloat(countStr, 64); err == nil {
				nodeGpuCount += n
			} else {
				log.Warnf("Could not parse GPU count from '%s' in GRES entry '%s'", countStr, entry)
			}
		}

		if nodeGpuCount > 0 {
			// A node can be in multiple partitions (comma-separated in sinfo output)
			nodePartitions := strings.Split(partitionNamesStr, ",")
			for _, pName := range nodePartitions {
				cleanPName := strings.TrimSuffix(pName, "*") // Remove asterisk from default partition names
				if cleanPName != "" {
					totals[cleanPName] += nodeGpuCount
				}
			}
		}
	}
	return totals
}

// parseAllocatedGPUsPerPartition fetches and parses the number of GPUs allocated to running jobs per partition.
func parseAllocatedGPUsPerPartition() map[string]float64 {
	alloc := make(map[string]float64)

	args := []string{
		"-a", "-X",
		"--format=Partition,AllocTRES",
		"--state=RUNNING",
		"--noheader", "--parsable2",
	}
	out := string(executeSlurmCommand("sacct", args))

	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		if line == "" {
			continue
		}
		col := strings.SplitN(line, "|", 2)
		if len(col) != 2 {
			continue
		}
		part, tres := col[0], col[1]

		var (
			jobGPUs      float64
			genericFound bool
		)

		for _, kv := range strings.Split(tres, ",") {
			kv = strings.ToLower(strings.TrimSpace(kv))
			if strings.HasPrefix(kv, "gres/gpu=") {
				// Generic entry ⇒ take it and ignore all type-specific ones.
				if v, err := strconv.ParseFloat(strings.SplitN(kv, "=", 2)[1], 64); err == nil {
					jobGPUs = v
					genericFound = true
				}
				break
			}
			if strings.HasPrefix(kv, "gres/gpu:") && !genericFound {
				if v, err := strconv.ParseFloat(strings.SplitN(kv, "=", 2)[1], 64); err == nil {
					jobGPUs += v
				}
			}
		}

		if jobGPUs > 0 && part != "" {
			alloc[part] += jobGPUs
		}
	}
	return alloc
}


// GetPartitionGPUMetrics collects and calculates GPU metrics for all partitions.
func GetPartitionGPUMetrics() map[string]*PartitionGPUMetrics {
	metricsMap := make(map[string]*PartitionGPUMetrics)
	totalGPUsPerPartition := parseTotalGPUsPerPartition()
	allocatedGPUsPerPartition := parseAllocatedGPUsPerPartition()

	allPartitionNames := make(map[string]bool)
	for p := range totalGPUsPerPartition {
		allPartitionNames[p] = true
	}
	for p := range allocatedGPUsPerPartition {
		allPartitionNames[p] = true
	}

	for partition := range allPartitionNames {
		total := totalGPUsPerPartition[partition] // Defaults to 0 if not present
		allocated := allocatedGPUsPerPartition[partition] // Defaults to 0 if not present

		// Ensure allocated does not exceed total for idle calculation sanity.
		// This can happen with complex GRES setups or slight timing differences in commands.
		// If total is 0 but GPUs are allocated (e.g. sinfo issue), log it.
		if allocated > total {
			log.Warnf("Partition %s: allocated GPUs (%f) > total GPUs (%f). Using allocated value as total for metric consistency, or capping idle at 0.", partition, allocated, total)
            // Option 1: Cap allocated at total for idle calculation
            // currentAllocatedForIdle := math.Min(allocated, total)
            // idle := total - currentAllocatedForIdle
            // Option 2: If total is unreliable and allocated is present, perhaps total should be at least allocated.
            // For now, report raw 'total' from sinfo, and calculate idle, ensuring it's not negative.
		}
        
        idle := total - allocated
        if idle < 0 {
            idle = 0 // Idle GPUs cannot be negative
        }


		// Only create metrics for partitions that meaningfully have GPUs or GPU allocations.
		if total > 0 || allocated > 0 {
			metricsMap[partition] = &PartitionGPUMetrics{
				allocated: allocated,
				idle:      idle,
				total:     total,
			}
		}
	}
	return metricsMap
}

// PartitionGPUsCollector implements the prometheus.Collector interface.
type PartitionGPUsCollector struct {
	allocated *prometheus.Desc
	idle      *prometheus.Desc
	total     *prometheus.Desc
}

// NewPartitionGPUsCollector creates a new PartitionGPUsCollector.
func NewPartitionGPUsCollector() *PartitionGPUsCollector {
	labels := []string{"partition"}
	return &PartitionGPUsCollector{
		allocated: prometheus.NewDesc("slurm_partition_gpus_allocated", "Allocated GPUs for partition", labels, nil),
		idle:      prometheus.NewDesc("slurm_partition_gpus_idle", "Idle GPUs for partition", labels, nil),
		total:     prometheus.NewDesc("slurm_partition_gpus_total", "Total GPUs for partition", labels, nil),
	}
}

// Describe sends the super-set of all possible descriptors of metrics collected by this Collector.
func (pgc *PartitionGPUsCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- pgc.allocated
	ch <- pgc.idle
	ch <- pgc.total
}

// Collect is called by the Prometheus registry when collecting metrics.
func (pgc *PartitionGPUsCollector) Collect(ch chan<- prometheus.Metric) {
	partitionMetrics := GetPartitionGPUMetrics()
	fmt.Println(partitionMetrics)
	for partitionName, data := range partitionMetrics {
		// Only send metrics if there's some GPU activity or configuration.
		if data.total > 0 || data.allocated > 0 {
			ch <- prometheus.MustNewConstMetric(pgc.allocated, prometheus.GaugeValue, data.allocated, partitionName)
			ch <- prometheus.MustNewConstMetric(pgc.idle, prometheus.GaugeValue, data.idle, partitionName)
			ch <- prometheus.MustNewConstMetric(pgc.total, prometheus.GaugeValue, data.total, partitionName)
		}
	}
}