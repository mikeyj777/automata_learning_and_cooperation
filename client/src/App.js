import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './components/Home';
import Ps001 from './components/Ps001';
import Ps001Controls from './components/ui/Ps001Controls';
import ProblemLayout from './components/ui/ProblemLayout';
import './App.css';
import './styles/global.css';

const App = () => {
  return (
    <ProblemLayout
      problemContent={<Ps001 />}
      problemControls={<Ps001Controls />}
    />
  )
};

export default App;