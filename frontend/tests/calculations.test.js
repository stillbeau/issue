import test from 'node:test';
import assert from 'node:assert';
import {
  plantCostPerSqft,
  variancePerSqft,
  overspendTotal,
  underspendTotal,
  reworkPctOfCost,
  isInvoiceStale
} from '../dist/src/calculations.js';

test('variance and spend calculations', () => {
  const pcs = plantCostPerSqft(3000, 100);
  const variance = variancePerSqft(pcs, 26.37);
  assert.ok(Math.abs(pcs - 30) < 0.001);
  assert.ok(Math.abs(variance - 3.63) < 0.01);
  assert.ok(Math.abs(overspendTotal(variance, 100) - 363) < 0.01);
  assert.strictEqual(underspendTotal(variance, 100), 0);
});

test('invoice aging check', () => {
  assert.strictEqual(isInvoiceStale('Pending', '2024-01-01', new Date('2024-03-15')), true);
  assert.strictEqual(isInvoiceStale('Pending', '2024-03-01', new Date('2024-03-15')), false);
  assert.strictEqual(isInvoiceStale('Paid', '2024-01-01', new Date('2024-03-15')), false);
});

test('rework percentage', () => {
  assert.strictEqual(reworkPctOfCost(200, 4000), 0.05);
  assert.strictEqual(reworkPctOfCost(null, 4000), null);
});
