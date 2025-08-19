"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.isInvoiceStale = exports.reworkPctOfCost = exports.underspendTotal = exports.overspendTotal = exports.variancePerSqft = exports.plantCostPerSqft = void 0;
const plantCostPerSqft = (plantInvoice, sqft) => {
    return sqft === 0 ? 0 : plantInvoice / sqft;
};
exports.plantCostPerSqft = plantCostPerSqft;
const variancePerSqft = (costPerSqft, target) => {
    return costPerSqft - target;
};
exports.variancePerSqft = variancePerSqft;
const overspendTotal = (variance, sqft) => {
    return variance > 0 ? variance * sqft : 0;
};
exports.overspendTotal = overspendTotal;
const underspendTotal = (variance, sqft) => {
    return variance < 0 ? Math.abs(variance) * sqft : 0;
};
exports.underspendTotal = underspendTotal;
const reworkPctOfCost = (reworkCost, totalCost) => {
    if (reworkCost == null || totalCost == null || totalCost === 0)
        return null;
    return reworkCost / totalCost;
};
exports.reworkPctOfCost = reworkPctOfCost;
const isInvoiceStale = (status, dateString, today = new Date()) => {
    if (!status || status.toLowerCase() !== 'pending' || !dateString)
        return false;
    const invoiceDate = new Date(dateString);
    const diff = today.getTime() - invoiceDate.getTime();
    const days = diff / (1000 * 60 * 60 * 24);
    return days > 30;
};
exports.isInvoiceStale = isInvoiceStale;
