"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const node_test_1 = __importDefault(require("node:test"));
const node_assert_1 = __importDefault(require("node:assert"));
const calculations_1 = require("../src/calculations");
(0, node_test_1.default)('variance and spend calculations', () => {
    const pcs = (0, calculations_1.plantCostPerSqft)(3000, 100);
    const variance = (0, calculations_1.variancePerSqft)(pcs, 26.37);
    node_assert_1.default.ok(Math.abs(pcs - 30) < 0.001);
    node_assert_1.default.ok(Math.abs(variance - 3.63) < 0.01);
    node_assert_1.default.strictEqual((0, calculations_1.overspendTotal)(variance, 100), 363);
    node_assert_1.default.strictEqual((0, calculations_1.underspendTotal)(variance, 100), 0);
});
(0, node_test_1.default)('invoice aging check', () => {
    node_assert_1.default.strictEqual((0, calculations_1.isInvoiceStale)('Pending', '2024-01-01', new Date('2024-03-15')), true);
    node_assert_1.default.strictEqual((0, calculations_1.isInvoiceStale)('Pending', '2024-03-01', new Date('2024-03-15')), false);
    node_assert_1.default.strictEqual((0, calculations_1.isInvoiceStale)('Paid', '2024-01-01', new Date('2024-03-15')), false);
});
(0, node_test_1.default)('rework percentage', () => {
    node_assert_1.default.strictEqual((0, calculations_1.reworkPctOfCost)(200, 4000), 0.05);
    node_assert_1.default.strictEqual((0, calculations_1.reworkPctOfCost)(null, 4000), null);
});
