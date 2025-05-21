package main

import (
	"strconv"
	"strings"
	"github.com/prometheus/client_golang/prometheus"
)

// parseAllocatedGPUsPerUser returns a map[user]allocatedGPUs for running jobs.
func parseAllocatedGPUsPerUser() map[string]float64 {
	userAlloc := make(map[string]float64)

	args := []string{
		"-a", "-X",
		"--format=User,AllocTRES", // <-- group on User instead of Partition
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
		user, tres := cols[0], cols[1]

		var (
			jobGPUs      float64
			genericFound bool
		)

		for _, kv := range strings.Split(tres, ",") {
			kv = strings.ToLower(strings.TrimSpace(kv))
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

		if jobGPUs > 0 && user != "" {
			userAlloc[user] += jobGPUs
		}
	}
	return userAlloc
}

// GetUserGPUMetrics calls the parser and reshapes the map for the collector.
func GetUserGPUMetrics() map[string]float64 {
	return parseAllocatedGPUsPerUser()
}

// Prometheus collector
type UserGPUsCollector struct {
	allocated *prometheus.Desc
}

// NewUserGPUsCollector creates a collector with a single "user" label.
func NewUserGPUsCollector() *UserGPUsCollector {
	return &UserGPUsCollector{
		allocated: prometheus.NewDesc(
			"slurm_user_gpus_allocated",
			"Allocated GPUs per Slurm user",
			[]string{"user"}, nil,
		),
	}
}

func (ugc *UserGPUsCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- ugc.allocated
}

func (ugc *UserGPUsCollector) Collect(ch chan<- prometheus.Metric) {
	userMetrics := GetUserGPUMetrics()
	for user, alloc := range userMetrics {
		ch <- prometheus.MustNewConstMetric(
			ugc.allocated,
			prometheus.GaugeValue,
			alloc,
			user,
		)
	}
}