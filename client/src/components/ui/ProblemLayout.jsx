import React, { useState, useEffect } from 'react';

const ProblemLayout = ({ 
  problemContent,
  problemControls
}) => {
  const [isRunning, setIsRunning] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [resetGrid, setResetGrid] = useState(false);
  const [probControlsDict, setProbControlsDict] = useState({
    targetPattern: 'glider',
    gridSize: 100
  });

  useEffect(() => {
    let interval;
    if (isRunning) {
      interval = setInterval(() => {
        setEpoch(prev => prev + 1);
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRunning]);

  const handleToggle = () => {
    setIsRunning(!isRunning);
  };

  const handleReset = () => {
    setIsRunning(false);
    setEpoch(0);
    setResetGrid(true);
  };

  return (
    <div className="problem-layout">
      <div className="main-panel">
        {React.isValidElement(problemContent) 
          ? React.cloneElement(problemContent, { isRunning, epoch, resetGrid, setResetGrid, probControlsDict})
          : problemContent
        }
      </div>
      
      <div className="right-container">
        <div className="controls-panel">
          <button 
            className={`toggle-button ${isRunning ? 'running' : 'stopped'}`}
            onClick={handleToggle}
          >
            {isRunning ? 'Stop' : 'Start'}
          </button>
          
          <button 
            className="reset-button"
            onClick={handleReset}
          >
            Reset
          </button>
          
          <div className="epoch-display">
            <span className="epoch-label">Epoch:</span>
            <span className="epoch-value">{epoch}</span>
          </div>
        </div>
        
        <div className="additional-controls-panel">
          {React.isValidElement(problemControls)
            ? React.cloneElement(problemControls, { isRunning, epoch, onReset: handleReset, controlsState:probControlsDict, updateControls: setProbControlsDict })
            : problemControls
          }
        </div>
      </div>
    </div>
  );
};

export default ProblemLayout;