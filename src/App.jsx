import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Login from "./Detect/Login";
import SafetyMonitoringSystem from "./Detect/SafetyMonitoring";

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/safety-monitoring-system" element={<SafetyMonitoringSystem />} />
      </Routes>
    </Router>
  );
};

export default App;