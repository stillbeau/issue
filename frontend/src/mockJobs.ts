export interface Job {
  jobName: string;
  account: string;
  salesperson: string;
  sqft?: number;
  sellPrice?: number;
  plantInvoice?: number;
  laborCost?: number;
  otherCogs?: number;
  reworkCost?: number;
  totalJobCost?: number;
  invoiceStatus?: string;
  invoiceDate?: string;
}

export const mockJobs: Job[] = [
  {
    jobName: 'Healthy',
    account: 'Alpha',
    salesperson: 'Alice',
    sqft: 100,
    sellPrice: 5000,
    plantInvoice: 2500,
    laborCost: 1000,
    otherCogs: 500,
    reworkCost: 0,
    totalJobCost: 4000,
    invoiceStatus: 'Paid',
    invoiceDate: '2024-04-01'
  },
  {
    jobName: 'Overspend',
    account: 'Bravo',
    salesperson: 'Bob',
    sqft: 120,
    sellPrice: 6000,
    plantInvoice: 4000,
    laborCost: 1200,
    otherCogs: 600,
    reworkCost: 0,
    totalJobCost: 5800,
    invoiceStatus: 'Paid',
    invoiceDate: '2024-03-01'
  },
  {
    jobName: 'Underspend',
    account: 'Charlie',
    salesperson: 'Carol',
    sqft: 90,
    sellPrice: 4500,
    plantInvoice: 1800,
    laborCost: 900,
    otherCogs: 400,
    reworkCost: 0,
    totalJobCost: 3100,
    invoiceStatus: 'Pending',
    invoiceDate: '2024-02-01'
  },
  {
    jobName: 'Missing Invoice',
    account: 'Delta',
    salesperson: 'Dan',
    sqft: 80,
    sellPrice: 3200,
    plantInvoice: undefined,
    laborCost: 800,
    otherCogs: 300,
    reworkCost: 0,
    totalJobCost: 0,
    invoiceStatus: 'Pending',
    invoiceDate: undefined
  },
  {
    jobName: 'Stale Invoice',
    account: 'Echo',
    salesperson: 'Eve',
    sqft: 110,
    sellPrice: 5500,
    plantInvoice: 3300,
    laborCost: 1100,
    otherCogs: 550,
    reworkCost: 0,
    totalJobCost: 4950,
    invoiceStatus: 'Pending',
    invoiceDate: '2024-01-01'
  },
  {
    jobName: 'Rework',
    account: 'Foxtrot',
    salesperson: 'Frank',
    sqft: 95,
    sellPrice: 4800,
    plantInvoice: 2600,
    laborCost: 950,
    otherCogs: 450,
    reworkCost: 200,
    totalJobCost: 4000,
    invoiceStatus: 'Paid',
    invoiceDate: '2024-04-15'
  }
];
