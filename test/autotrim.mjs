import fs from 'node:fs';
import { suggestCuts } from '../js/core.js';
const [base, plan] = process.argv.slice(2);
const times = JSON.parse(fs.readFileSync(base + '.json')).times, p = JSON.parse(fs.readFileSync(plan));
const segs = suggestCuts(times, p.sharp, p.ok, p.C);
const kept = segs.reduce((a, s) => a + s.end - s.start, 0);
console.log('suggested keeps:', segs.map(s => `${s.start.toFixed(1)}–${s.end.toFixed(1)}s`).join(', '), `| kept ${kept.toFixed(1)} of ${times[times.length - 1].toFixed(1)} s`);
