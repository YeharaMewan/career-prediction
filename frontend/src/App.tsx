import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import MainLayout from "./Layouts/MainLayout";
import Landing from "./pages/home/section/Landing";
import Prediction from "./pages/predictionready";
import FAQ from "./pages/faq/section/FAQ";
import CareerPath from "./pages/explorecareerPath/careerPath";

function App() {
  return (
    <Router>
      <MainLayout>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/prediction" element={<Prediction />} />
          <Route path="/faq" element={<FAQ />} />
          <Route path="/careerPath" element={<CareerPath />} />
        </Routes>
      </MainLayout>
    </Router>
  );
}

export default App;
