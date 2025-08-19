export const plantCostPerSqft = (plantInvoice: number, sqft: number): number => {
  return sqft === 0 ? 0 : plantInvoice / sqft;
};

export const variancePerSqft = (costPerSqft: number, target: number): number => {
  return costPerSqft - target;
};

export const overspendTotal = (variance: number, sqft: number): number => {
  return variance > 0 ? variance * sqft : 0;
};

export const underspendTotal = (variance: number, sqft: number): number => {
  return variance < 0 ? Math.abs(variance) * sqft : 0;
};

export const reworkPctOfCost = (reworkCost: number | null, totalCost: number | null): number | null => {
  if (reworkCost == null || totalCost == null || totalCost === 0) return null;
  return reworkCost / totalCost;
};

export const isInvoiceStale = (status: string | null, dateString: string | null, today: Date = new Date()): boolean => {
  if (!status || status.toLowerCase() !== 'pending' || !dateString) return false;
  const invoiceDate = new Date(dateString);
  const diff = today.getTime() - invoiceDate.getTime();
  const days = diff / (1000 * 60 * 60 * 24);
  return days > 30;
};
