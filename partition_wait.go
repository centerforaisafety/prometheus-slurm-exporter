package main

import (
	"bufio"
	"context"
	"os/exec"
	"sort"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/common/log"
)

const layoutISO = "2006-01-02T15:04:05" // Slurm’s %V default format

type PartitionWaitCollector struct {
	descAvg    *prometheus.Desc
	descMedian *prometheus.Desc
	timeout    time.Duration
	clock      func() time.Time
}

func NewPartitionWaitCollector() prometheus.Collector {
	return &PartitionWaitCollector{
		descAvg: prometheus.NewDesc(
			"slurm_partition_pending_wait_seconds_avg",
			"Average wait (seconds) of currently PENDING jobs, grouped by partition.",
			[]string{"partition"}, nil,
		),
		descMedian: prometheus.NewDesc(
			"slurm_partition_pending_wait_seconds_median",
			"Median wait (seconds) of currently PENDING jobs, grouped by partition.",
			[]string{"partition"}, nil,
		),
		timeout: 5 * time.Second,
		clock:   time.Now,
	}
}

func (c *PartitionWaitCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.descAvg
	ch <- c.descMedian
}

func (c *PartitionWaitCollector) Collect(ch chan<- prometheus.Metric) {
	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "squeue",
		"-h",          // no header
		"--states=PD", // pending only
		"--format=%P|%V",
	)

	out, err := cmd.Output()
	if err != nil {
		log.Errorf("partition_wait_collector: squeue failed: %v", err)
		return
	}

	type stats struct {
		sum   float64
		times []float64 // individual waits for median
	}

	partitions := make(map[string]*stats)
	now := c.clock()

	scanner := bufio.NewScanner(strings.NewReader(string(out)))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "|", 2)
		if len(parts) != 2 {
			log.Warnf("partition_wait_collector: bad line %q", line)
			continue
		}
		part, ts := parts[0], parts[1]

		submit, err := time.ParseInLocation(layoutISO, ts, time.Local)
		if err != nil {
			log.Warnf("partition_wait_collector: cannot parse time %q: %v", ts, err)
			continue
		}

		wait := now.Sub(submit).Seconds()
		if wait < 0 { // clock skew safety
			wait = 0
		}

		s, ok := partitions[part]
		if !ok {
			s = &stats{}
			partitions[part] = s
		}
		s.sum += wait
		s.times = append(s.times, wait)
	}
	if err := scanner.Err(); err != nil {
		log.Errorf("partition_wait_collector: scanner: %v", err)
	}

	for part, s := range partitions {
		if len(s.times) == 0 {
			continue
		}

		// Average
		avg := s.sum / float64(len(s.times))
		if m, err := prometheus.NewConstMetric(c.descAvg, prometheus.GaugeValue, avg, part); err == nil {
			ch <- m
		}

		// Median
		sort.Float64s(s.times)
		var median float64
		n := len(s.times)
		if n%2 == 1 {
			median = s.times[n/2]
		} else {
			median = (s.times[n/2-1] + s.times[n/2]) / 2
		}
		if m, err := prometheus.NewConstMetric(c.descMedian, prometheus.GaugeValue, median, part); err == nil {
			ch <- m
		}
	}
}
