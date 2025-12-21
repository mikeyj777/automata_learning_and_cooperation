import React, { useState, useEffect } from 'react';

// Custom controls component for the bottom right panel
// Receives isRunning, epoch, and onReset from ProblemLayout
const Ps001Controls = ({ isRunning, epoch, onReset, controlsState, updateControls }) => {
  const {targetPattern, gridSize} = controlsState;

  const handlePatternChange = (e) => {
    updateControls(prev => ({ ...prev, targetPattern: e.target.value }));
  };

  const handleGridSizeChange = (e) => {
    updateControls(prev => ({ ...prev, gridSize: parseInt(e.target.value) }));
  };

  const patterns = [
    { value: 'glider', label: 'Glider' },
    { value: 'gosper_glider_gun', label: 'Gosper Glider Gun' },
    { value: 'pulsar', label: 'Pulsar' },
    { value: 'lwss', label: 'Lightweight Spaceship' },
    { value: 'blinker', label: 'Blinker' },
    { value: 'toad', label: 'Toad' },
    { value: 'beacon', label: 'Beacon' },
    { value: 'pentadecathlon', label: 'Pentadecathlon' },
    { value: 'r_pentomino', label: 'R-Pentomino' },
    { value: 'diehard', label: 'Diehard' },
    { value: 'acorn', label: 'Acorn' }
  ];

  return (
    <div>
      <h3>Problem Controls</h3>
      
      <div className="control-group">
        <label className="control-label">
          Target Pattern
        </label>
        <select
          value={targetPattern}
          onChange={handlePatternChange}
          disabled={isRunning}
          className="control-select"
        >
          {patterns.map(pattern => (
            <option key={pattern.value} value={pattern.value}>
              {pattern.label}
            </option>
          ))}
        </select>
      </div>

      <div className="control-group">
        <label className="control-label">
          Grid Size
        </label>
        <input 
          type="number"
          min="10"
          max="200"
          step="10"
          value={gridSize}
          onChange={handleGridSizeChange}
          disabled={isRunning}
          className="control-input"
        />
      </div>

      <div className="status-display">
        <p><strong>Status:</strong> {isRunning ? 'Running' : 'Idle'}</p>
        <p><strong>Pattern:</strong> {patterns.find(p => p.value === targetPattern)?.label}</p>
        <p><strong>Grid:</strong> {gridSize} × {gridSize}</p>
      </div>
    </div>
  );
};

export default Ps001Controls;