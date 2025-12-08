import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import Button from "../../components/Button";
import Graph from "../../components/Graph";
import Border from "../../components/border";
import PredictionSkeleton from "../../components/skeletons/PredictionSkeleton";
import type { CareerPrediction } from "../../types/career";

// Color palette for career cards (same as careerPath.tsx)
const CARD_COLORS = ["#006466", "#002657", "#7c162e", "#4f0147", "#2d1805"];

function Prediction() {
  const location = useLocation();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(true);

  // Get predictions AND sessionId from navigation state
  const predictions = (location.state?.predictions || []) as CareerPrediction[];
  const sessionId = location.state?.sessionId;
  const language = location.state?.language || 'en';

  useEffect(() => {
    // Redirect to chat if no predictions available
    if (predictions.length === 0) {
      navigate("/chat");
      return;
    }

    // Simulate loading for smooth UX
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 700);

    return () => clearTimeout(timer);
  }, [predictions, navigate]);

  if (isLoading) {
    return <PredictionSkeleton />;
  }

  return (
    <div className="p-10 flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-teal-50 via-cyan-50 to-teal-100">
      <h1 className="text-4xl font-bold text-center mb-8 bg-gradient-to-r from-teal-600 to-cyan-700 bg-clip-text text-transparent">
        Your Career Recommendations
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-16 w-full max-w-7xl">
        {predictions.map((career, index) => {
          const matchScore = Math.round(career.confidence_score * 100);
          const cardColor = CARD_COLORS[index % CARD_COLORS.length];

          const isLastAndOdd = predictions.length % 2 !== 0 && index === predictions.length - 1;

          return (
            <div
              key={index}
              className={isLastAndOdd ? "md:col-span-2 flex justify-center" : ""}
            >
              <div className={isLastAndOdd ? "w-full md:w-[calc(50%-1rem)]" : "w-full"}>
                <Border borderRadius="1rem" color={cardColor}>
                  <div className="flex flex-row">
                    <div className="w-2/3">
                      <h2
                        className="font-mono uppercase font-semibold text-2xl"
                        style={{ color: cardColor }}
                      >
                        {career.title}
                      </h2>
                      <div className="flex flex-col mt-3">
                        <p className="font-sans" style={{ color: cardColor, opacity: 0.8 }}>
                          {career.description}
                        </p>
                        {/* Removed skills section - showing only title, description, match score */}
                        <div className="mt-6">
                          <Button
                            label="Select this path"
                            className="uppercase font-mono text-white"
                            style={{ backgroundColor: cardColor }}
                            onClick={() => {
                              // Navigate to loading page, which triggers career planning
                              navigate('/career-planning-loader', {
                                state: {
                                  sessionId: sessionId,
                                  careerTitle: career.title,
                                  language: language,
                                },
                              });
                            }}
                          />
                        </div>
                      </div>
                    </div>
                    <div className="p-2 flex justify-center items-center">
                      <Graph
                        value={matchScore}
                        label="MATCH"
                        strokeColor={cardColor}
                      />
                    </div>
                  </div>
                </Border>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default Prediction;
