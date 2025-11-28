import { useState, useEffect } from "react";
import RotatingCards from "../../components/RotatingCards";
import PredictionSkeleton from "../../components/skeletons/PredictionSkeleton";

function Prediction() {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate content loading
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 700);

    return () => clearTimeout(timer);
  }, []);

  if (isLoading) {
    return <PredictionSkeleton />;
  }

  return (
    <div className="flex h-screen items-center justify-center">
      <RotatingCards />
    </div>
  );
}
export default Prediction;
