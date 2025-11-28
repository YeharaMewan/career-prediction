import { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import MainLayout from "./Layouts/MainLayout";
import Landing from "./pages/home/section/Landing";
import Prediction from "./pages/predictionready";
import FAQ from "./pages/faq/section/FAQ";
import SplashScreen from "./components/SplashScreen";
import { LanguageProvider } from "./context/LanguageContext";

import ChatPage from "./pages/chat/ChatPage";
import CareerPath from "./pages/explorecareerPath/careerPath";

function App() {
  const [showSplash, setShowSplash] = useState(true);
  const [isAppReady, setIsAppReady] = useState(false);

  useEffect(() => {
    // Simulate app initialization (fonts, assets, etc.)
    const timer = setTimeout(() => {
      setIsAppReady(true);
    }, 500);

    return () => clearTimeout(timer);
  }, []);

  const handleSplashComplete = () => {
    setShowSplash(false);
  };

  return (
    <LanguageProvider>
      <AnimatePresence mode="wait">
        {showSplash && isAppReady && (
          <SplashScreen onComplete={handleSplashComplete} />
        )}
      </AnimatePresence>

      {!showSplash && (
        <Router>
          <MainLayout>
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/prediction" element={<Prediction />} />
              <Route path="/chat" element={<ChatPage />} />
              {/* Add more routes as needed */}
              <Route path="/faq" element={<FAQ />} />
              <Route path="/careerPath" element={<CareerPath />} />
            </Routes>
          </MainLayout>
        </Router>
      )}
    </LanguageProvider>

  );
}

export default App;
