import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Rate, Counter } from 'k6/metrics';
import { SharedArray } from 'k6/data';

const p50Trend = new Trend('request_duration_p50');
const p95Trend = new Trend('request_duration_p95');
const p99Trend = new Trend('request_duration_p99');
const errorRate = new Rate('error_rate');
const approvedRate = new Rate('approved_rate');
const requestCount = new Counter('request_count');

const payloads = new SharedArray('payloads', function () {
  return JSON.parse(open('./test-payloads.json'));
});

export const options = {
  scenarios: {
    ramp_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 20 },
        { duration: '60s', target: 20 },
        { duration: '30s', target: 50 },
        { duration: '60s', target: 50 },
        { duration: '30s', target: 100 },
        { duration: '60s', target: 100 },
        { duration: '30s', target: 0 },
      ],
    },
  },
  thresholds: {
    http_req_duration: ['p(50)<100', 'p(95)<300', 'p(99)<500'],
    http_req_failed: ['rate<0.01'],
    error_rate: ['rate<0.01'],
  },
  summaryTrendStats: ['min', 'avg', 'med', 'p(50)', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

const params = {
  headers: {
    'Content-Type': 'application/json',
  },
};

export default function () {
  const payload = payloads[Math.floor(Math.random() * payloads.length)];
  const body = JSON.stringify(payload);

  const res = http.post('http://localhost:9999/fraud-score', body, params);

  requestCount.add(1);

  const duration = res.timings.duration;
  p50Trend.add(duration);
  p95Trend.add(duration);
  p99Trend.add(duration);

  const failed = res.status < 200 || res.status >= 400;
  errorRate.add(failed);

  const checkResult = check(res, {
    'status is 200': (r) => r.status === 200,
    'response is JSON': (r) => {
      try {
        const data = r.json();
        return typeof data.approved === 'boolean' && typeof data.fraud_score === 'number';
      } catch (e) {
        return false;
      }
    },
  });

  if (checkResult) {
    try {
      approvedRate.add(res.json().approved === true);
    } catch (e) {
      // ignore parse errors
    }
  }

  if (failed) {
    errorRate.add(1);
  }

  sleep(0.1);
}
