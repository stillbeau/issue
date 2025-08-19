"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const react_1 = __importDefault(require("react"));
const mockJobs_1 = require("./mockJobs");
const calculations_1 = require("./calculations");
const TARGET = 26.37;
const ProductionMarginsTab = () => {
    const jobs = mockJobs_1.mockJobs;
    const totals = jobs.reduce((acc, job) => {
        if (!job.sqft || !job.plantInvoice) {
            acc.missing += 1;
            return acc;
        }
        const pcs = (0, calculations_1.plantCostPerSqft)(job.plantInvoice, job.sqft);
        const variance = (0, calculations_1.variancePerSqft)(pcs, TARGET);
        acc.plantCostSum += pcs;
        acc.overspend += (0, calculations_1.overspendTotal)(variance, job.sqft);
        acc.underspend += (0, calculations_1.underspendTotal)(variance, job.sqft);
        if (variance > 0)
            acc.overTarget += 1;
        return acc;
    }, { plantCostSum: 0, overspend: 0, underspend: 0, overTarget: 0, missing: 0 });
    const avgPlantCost = jobs.length ? totals.plantCostSum / jobs.length : 0;
    const percentOverTarget = jobs.length ? (totals.overTarget / jobs.length) * 100 : 0;
    return (react_1.default.createElement("div", null,
        react_1.default.createElement("h2", null, "Production Margins"),
        react_1.default.createElement("div", null,
            react_1.default.createElement("div", null,
                "Average Plant Cost / Sq Ft: $",
                avgPlantCost.toFixed(2)),
            react_1.default.createElement("div", null,
                "% of Jobs Over Target: ",
                percentOverTarget.toFixed(1),
                "%"),
            react_1.default.createElement("div", null,
                "Total Overspend $: $",
                totals.overspend.toFixed(2)),
            react_1.default.createElement("div", null,
                "Total Underspend $: $",
                totals.underspend.toFixed(2)),
            react_1.default.createElement("div", null,
                "Missing Data Count: ",
                totals.missing)),
        react_1.default.createElement("table", null,
            react_1.default.createElement("thead", null,
                react_1.default.createElement("tr", null,
                    react_1.default.createElement("th", null, "Job Name"),
                    react_1.default.createElement("th", null, "Sq Ft"),
                    react_1.default.createElement("th", null, "Plant Invoice"),
                    react_1.default.createElement("th", null, "Plant $/SqFt"),
                    react_1.default.createElement("th", null, "Variance"),
                    react_1.default.createElement("th", null, "Overspend"),
                    react_1.default.createElement("th", null, "Underspend"),
                    react_1.default.createElement("th", null, "Invoice Status"),
                    react_1.default.createElement("th", null, "Invoice Date"),
                    react_1.default.createElement("th", null, "Rework % of Cost"))),
            react_1.default.createElement("tbody", null, jobs.map((job) => {
                var _a, _b, _c;
                const pcs = job.plantInvoice && job.sqft ? (0, calculations_1.plantCostPerSqft)(job.plantInvoice, job.sqft) : 0;
                const variance = pcs ? (0, calculations_1.variancePerSqft)(pcs, TARGET) : 0;
                const overspend = (0, calculations_1.overspendTotal)(variance, job.sqft || 0);
                const underspend = (0, calculations_1.underspendTotal)(variance, job.sqft || 0);
                const reworkPct = (0, calculations_1.reworkPctOfCost)((_a = job.reworkCost) !== null && _a !== void 0 ? _a : null, (_b = job.totalJobCost) !== null && _b !== void 0 ? _b : null);
                const stale = (0, calculations_1.isInvoiceStale)(job.invoiceStatus || null, job.invoiceDate || null);
                return (react_1.default.createElement("tr", { key: job.jobName, style: { color: stale ? 'red' : undefined } },
                    react_1.default.createElement("td", null, job.jobName),
                    react_1.default.createElement("td", null, (_c = job.sqft) !== null && _c !== void 0 ? _c : '-'),
                    react_1.default.createElement("td", null, job.plantInvoice ? `$${job.plantInvoice.toFixed(2)}` : '-'),
                    react_1.default.createElement("td", null, pcs ? `$${pcs.toFixed(2)}` : '-'),
                    react_1.default.createElement("td", null, variance ? `$${variance.toFixed(2)}` : '-'),
                    react_1.default.createElement("td", null, overspend ? `$${overspend.toFixed(2)}` : '-'),
                    react_1.default.createElement("td", null, underspend ? `$${underspend.toFixed(2)}` : '-'),
                    react_1.default.createElement("td", null, job.invoiceStatus || '-'),
                    react_1.default.createElement("td", null, job.invoiceDate || '-'),
                    react_1.default.createElement("td", null, reworkPct !== null ? `${(reworkPct * 100).toFixed(1)}%` : '-')));
            })))));
};
exports.default = ProductionMarginsTab;
