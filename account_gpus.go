package main

import (
	"strconv"
	"strings"
	"github.com/prometheus/client_golang/prometheus"
)

// parseAllocatedGPUsPerAccount returns GPUs allocated to *running* jobs
// aggregated by Slurm account.
//
// sacct is used because it reports already-started jobs (unlike squeue,
// which can include pending jobs).  The --parsable2 output makes field
// splitting trivial.
func parseAllocatedGPUsPerAccount() map[string]float64 {
	alloc := make(map[string]float64)

	args := []string{
		"-a", "-X",
		"--format=Account,AllocTRES",
		"--state=RUNNING",
		"--noheader", "--parsable2",
	}

	out := string(executeSlurmCommand("sacct", args))

	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		if line == "" {
			continue
		}
		cols := strings.SplitN(line, "|", 2)
		if len(cols) != 2 {
			continue
		}
		acct, tres := cols[0], cols[1]

		var (
			jobGPUs      float64
			genericFound bool
		)

		for _, kv := range strings.Split(tres, ",") {
			kv = strings.ToLower(strings.TrimSpace(kv))

			// Generic “gres/gpu=” overrides type-specific counts.
			if strings.HasPrefix(kv, "gres/gpu=") {
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

		if jobGPUs > 0 && acct != "" {
			alloc[acct] += jobGPUs
		}
	}
	return alloc
}

// Metric container
type AccountGPUMetrics struct {
	allocated float64
}

// GetAccountGPUMetrics builds a map keyed by account.
func GetAccountGPUMetrics() map[string]*AccountGPUMetrics {
	allocPerAcct := parseAllocatedGPUsPerAccount()

	metrics := make(map[string]*AccountGPUMetrics, len(allocPerAcct))
	for acct, alloc := range allocPerAcct {
		metrics[acct] = &AccountGPUMetrics{allocated: alloc}
	}
	return metrics
}

// Prometheus collector
type AccountGPUsCollector struct {
	allocated *prometheus.Desc
}

// NewAccountGPUsCollector registers the “slurm_account_gpus_allocated” gauge.
func NewAccountGPUsCollector() *AccountGPUsCollector {
	return &AccountGPUsCollector{
		allocated: prometheus.NewDesc(
			"slurm_account_gpus_allocated",
			"GPUs allocated to running jobs per Slurm account",
			[]string{"account"},
			nil,
		),
	}
}

// Describe implements the prometheus.Collector interface.
func (c *AccountGPUsCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.allocated
}

// Collect gathers metrics.
func (c *AccountGPUsCollector) Collect(ch chan<- prometheus.Metric) {
	for acct, data := range GetAccountGPUMetrics() {
		// Skip accounts without running GPU jobs to avoid noisy time-series.
		if data.allocated > 0 {
			ch <- prometheus.MustNewConstMetric(
				c.allocated, prometheus.GaugeValue, data.allocated, acct,
			)
		}
	}
}
