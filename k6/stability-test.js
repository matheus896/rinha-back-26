import http from 'k6/http';
import { check } from 'k6';
import { Trend, Rate, Counter } from 'k6/metrics';
import { SharedArray } from 'k6/data';

const p99Trend = new Trend('request_duration_p99');
const p95Trend = new Trend('request_duration_p95');
const p50Trend = new Trend('request_duration_p50');
const errorRate = new Rate('errors');

const payloads = new SharedArray('payloads', function () {
  return JSON.parse(open('/test/test-payloads.json'));
});

export const options = {
  scenarios: {
    sustained: {
      executor: 'constant-arrival-rate',
      rate: 200,
      timeUnit: '1s',
      duration: '3m',
      preAllocatedVUs: 20,
      maxVUs: 60,
    },
  },
  thresholds: {
    'request_duration_p99': ['p(99)<2000'],
    'request_duration_p95': ['p(95)<500'],
    errors: ['rate<0.15'],
  },
  summaryTrendStats: ['min', 'avg', 'med', 'p(50)', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

const params = {
  headers: { 'Content-Type': 'application/json' },
};

export default function () {
  const payload = payloads[Math.floor(Math.random() * payloads.length)];
  const body = JSON.stringify(payload);

  const res = http.post('http://localhost:9999/fraud-score', body, params);

  const duration = res.timings.duration;
  p50Trend.add(duration);
  p95Trend.add(duration);
  p99Trend.add(duration);

  const failed = res.status !== 200;
  errorRate.add(failed);

  check(res, {
    'status 200': (r) => r.status === 200,
    'is JSON': (r) => {
      try {
        const data = r.json();
        return typeof data.approved === 'boolean' && typeof data.fraud_score === 'number';
      } catch (_) { return false; }
    },
  });
}
