import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom'
import SignUp from './components/SignUp';
import Login from './components/LogIn';
import Dashboard from './components/Dashboard';
import DataAnalysis from '.components/DataAnalysis';

function App() {
  return (
    <Router>
      <Switch>
        <Route path='/signup' component={SignUp} />
        <Route path='/login' component={Login} />
        <Route path='/dashboard' component={Dashboard} />
        <Route path='/data-analysis' component={DataAnalysis} />
      </Switch>
    </Router>
  );
}

export default App;