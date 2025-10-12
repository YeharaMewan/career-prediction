import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import MainLayout from "./Layouts/MainLayout";
import Landing from "./pages/home/section/Landing";
import Prediction from "./pages/predictionready";

function App() {
  return (
    <Router>
      <MainLayout>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/prediction" element={<Prediction />} />
          {/* Add more routes as needed */}
          {/* <Route path="/about" element={<Aboutus />} /> */}
        </Routes>
      </MainLayout>
    </Router>
  );
}

export default App;
