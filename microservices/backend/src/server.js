'use strict';

const express = require('express');

const PORT = process.env.PORT || 4000;
const app = express();
app.use(express.json());

// Normalise Math.floor (any casing) -> Math.floor so JS can evaluate it,
// then strip it out to validate the remaining characters against the allowlist.
const SAFE_REMAINING = /^[\d\s+\-*/%().]*$/;

function normalise(expr) {
  return expr.replace(/Math\.floor/gi, 'Math.floor');
}

function evaluate(expr) {
  const cleaned = expr.trim();

  // Remove all Math.floor occurrences (case-insensitive) then check only
  // allowed characters remain: digits, operators, parens, dots, whitespace.
  const stripped = cleaned.replace(/Math\.floor/gi, '');
  if (!SAFE_REMAINING.test(stripped)) {
    const err = new Error('Expression contains disallowed characters');
    err.status = 400;
    throw err;
  }

  // eslint-disable-next-line no-eval
  const result = eval(normalise(cleaned));

  if (typeof result !== 'number' || !isFinite(result)) {
    const err = new Error('Expression did not evaluate to a finite number');
    err.status = 400;
    throw err;
  }

  return result;
}

// POST /eval — { "expression": "<string>" } → { "result": <number> }
app.post('/eval', (req, res) => {
  const { expression } = req.body;
  if (!expression || typeof expression !== 'string') {
    return res.status(400).json({ error: 'expression is required and must be a string' });
  }

  try {
    const result = evaluate(expression);
    res.json({ result });
  } catch (err) {
    res.status(err.status || 500).json({ error: err.message });
  }
});

// GET /wiki?title=<string>&position=<int> → { "word": "<string>" }
app.get('/wiki', async (req, res) => {
  const { title, position } = req.query;

  if (!title || typeof title !== 'string') {
    return res.status(400).json({ error: 'title is required' });
  }

  const pos = parseInt(position, 10);
  if (!Number.isInteger(pos) || pos < 1) {
    return res.status(400).json({ error: 'position must be a positive integer' });
  }

  try {
    const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`;
    const response = await fetch(url);
    if (!response.ok) {
      return res.status(502).json({ error: `Wikipedia returned ${response.status} for "${title}"` });
    }

    const data = await response.json();
    const words = data.extract.split(/\s+/);
    const index = pos - 1;

    if (index >= words.length) {
      return res.status(422).json({ error: `Position ${pos} out of range (article has ${words.length} words)` });
    }

    res.json({ word: words[index] });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Backend microservice listening on port ${PORT}`);
});
