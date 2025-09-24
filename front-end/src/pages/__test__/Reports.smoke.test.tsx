import { describe, test, expect } from 'vitest';
import React from 'react';
import { render, screen } from '@testing-library/react';
import Reports from '../Reports';


describe('Reports (smoke)', () => {
  test('renders heading and heatmap grid', () => {
    render(<Reports />);

    // Heading and subtitle
    expect(screen.getByRole('heading', { level: 2, name: /Reports/i })).toBeInTheDocument();
    expect(screen.getByText(/Basic heatmap template/i)).toBeInTheDocument();

    // Heatmap container
    expect(document.querySelector('.heatmap')).toBeInTheDocument();

    // Should render 5 rows
    const rows = document.querySelectorAll('.heatmap-row');
    expect(rows.length).toBe(5);

    // Each row should have 12 cells
    rows.forEach(row => {
      const cells = row.querySelectorAll('.heatcell');
      expect(cells.length).toBe(12);
    });
  });

  test('shows the correct label when filePage prop is provided', () => {
    const { rerender } = render(<Reports filePage="file1" />);
    expect(screen.getByText(/FILE ANALYSIS 1/i)).toBeInTheDocument();

    rerender(<Reports filePage="file2" />);
    expect(screen.getByText(/FILE ANALYSIS 2/i)).toBeInTheDocument();
  });
});
