import React from 'react';

// Individual problem component - receives isRunning, epoch, and onReset from ProblemLayout
const ExampleProblem = ({ isRunning, epoch, onReset }) => {
  // Use the props from ProblemLayout to drive your problem logic
  React.useEffect(() => {
    if (isRunning) {
      console.log('Problem is running, epoch:', epoch);
      // Add your epoch-based logic here
      // e.g., update neural network weights, run simulation step, etc.
    }
  }, [isRunning, epoch]);

  return (
    <div>
      <h1>Problem 1: Example Problem</h1>
      <p>Current status: {isRunning ? 'Running' : 'Stopped'}</p>
      <p>Epoch: {epoch}</p>
      
      {/* Your problem-specific content and answer components go here */}
      <div>
        <h2>Problem Description</h2>
        <p>Your problem description goes here...</p>
      </div>
      
      <div>
        <h2>Answer/Visualization</h2>
        <p>Your problem solution, graphs, or visualizations go here...</p>
      </div>
    </div>
  );
};

export default ExampleProblem;