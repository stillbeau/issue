import React from 'react';
import { Job, mockJobs } from './mockJobs';
import {
  plantCostPerSqft,
  variancePerSqft,
  overspendTotal,
  underspendTotal,
  isInvoiceStale,
  reworkPctOfCost
} from './calculations';

const TARGET = 26.37;

const ProductionMarginsTab = () => {
  const jobs = mockJobs;

  const totals = jobs.reduce(
    (acc, job) => {
      if (!job.sqft || !job.plantInvoice) {
        acc.missing += 1;
        return acc;
      }
      const pcs = plantCostPerSqft(job.plantInvoice, job.sqft);
      const variance = variancePerSqft(pcs, TARGET);
      acc.plantCostSum += pcs;
      acc.overspend += overspendTotal(variance, job.sqft);
      acc.underspend += underspendTotal(variance, job.sqft);
      if (variance > 0) acc.overTarget += 1;
      return acc;
    },
    { plantCostSum: 0, overspend: 0, underspend: 0, overTarget: 0, missing: 0 }
  );

  const avgPlantCost = jobs.length ? totals.plantCostSum / jobs.length : 0;
  const percentOverTarget = jobs.length ? (totals.overTarget / jobs.length) * 100 : 0;

  return (
    <div>
      <h2>Production Margins</h2>
      <div>
        <div>Average Plant Cost / Sq Ft: ${avgPlantCost.toFixed(2)}</div>
        <div>% of Jobs Over Target: {percentOverTarget.toFixed(1)}%</div>
        <div>Total Overspend $: ${totals.overspend.toFixed(2)}</div>
        <div>Total Underspend $: ${totals.underspend.toFixed(2)}</div>
        <div>Missing Data Count: {totals.missing}</div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Job Name</th>
            <th>Sq Ft</th>
            <th>Plant Invoice</th>
            <th>Plant $/SqFt</th>
            <th>Variance</th>
            <th>Overspend</th>
            <th>Underspend</th>
            <th>Invoice Status</th>
            <th>Invoice Date</th>
            <th>Rework % of Cost</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job: Job) => {
            const pcs = job.plantInvoice && job.sqft ? plantCostPerSqft(job.plantInvoice, job.sqft) : 0;
            const variance = pcs ? variancePerSqft(pcs, TARGET) : 0;
            const overspend = overspendTotal(variance, job.sqft || 0);
            const underspend = underspendTotal(variance, job.sqft || 0);
            const reworkPct = reworkPctOfCost(job.reworkCost ?? null, job.totalJobCost ?? null);
            const stale = isInvoiceStale(job.invoiceStatus || null, job.invoiceDate || null);
            return (
              <tr key={job.jobName} style={{ color: stale ? 'red' : undefined }}>
                <td>{job.jobName}</td>
                <td>{job.sqft ?? '-'}</td>
                <td>{job.plantInvoice ? `$${job.plantInvoice.toFixed(2)}` : '-'}</td>
                <td>{pcs ? `$${pcs.toFixed(2)}` : '-'}</td>
                <td>{variance ? `$${variance.toFixed(2)}` : '-'}</td>
                <td>{overspend ? `$${overspend.toFixed(2)}` : '-'}</td>
                <td>{underspend ? `$${underspend.toFixed(2)}` : '-'}</td>
                <td>{job.invoiceStatus || '-'}</td>
                <td>{job.invoiceDate || '-'}</td>
                <td>{reworkPct !== null ? `${(reworkPct * 100).toFixed(1)}%` : '-'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default ProductionMarginsTab;
