import React, { useEffect, useState } from "react";

const RotatingCards: React.FC = () => {
  const [rotationCount, setRotationCount] = useState(0);
  const [prediction, setPrediction] = useState<string | null>(null);
  useEffect(() => {
    if (rotationCount < 2) {
      const timer = setTimeout(() => {
        setRotationCount(rotationCount + 1);
      }, 3000); // rotation duration
      return () => clearTimeout(timer);
    } else if (rotationCount === 2 && !prediction) {
      const simulatedPrediction = "Your AI Prediction Result!";
      setPrediction(simulatedPrediction);
    }
  }, [rotationCount, prediction]);
  const displayText =
    rotationCount < 2 ? "Prediction is get Readying.... " : (prediction ?? "Loading...");
  return (
    <div className="wrapper">
      <div className="inner" style={{ ["--quantity" as string]: 10 } as React.CSSProperties}>
        <div
          className="font-fredoka text-[40px] font-bold text-white"
          style={{
            transform: `translateY(-120px) rotateY(${rotationCount * 180}deg)`,
            transition: "transform 1s ease-in-out",
          }}
        >
          {displayText}
        </div>
        <div
          className="card"
          style={
            {
              ["--index" as string]: 0,
              ["--color-card" as string]: "142, 249, 252",
            } as React.CSSProperties
          }
        >
          <div className="img"></div>
        </div>
        <div
          className="card"
          style={
            {
              ["--index" as string]: 1,
              ["--color-card" as string]: "142, 252, 204",
            } as React.CSSProperties
          }
        >
          <div className="img"></div>
        </div>
        <div
          className="card"
          style={
            {
              ["--index" as string]: 2,
              ["--color-card" as string]: "142, 252, 157",
            } as React.CSSProperties
          }
        >
          <div className="img"></div>
        </div>
        <div
          className="card"
          style={
            {
              ["--index" as string]: 3,
              ["--color-card" as string]: "215, 252, 142",
            } as React.CSSProperties
          }
        >
          <div className="img"></div>
        </div>
        <div
          className="card"
          style={
            {
              ["--index" as string]: 4,
              ["--color-card" as string]: "252, 252, 142",
            } as React.CSSProperties
          }
        >
          <div className="img"></div>
        </div>
        <div
          className="card"
          style={
            {
              ["--index" as string]: 5,
              ["--color-card" as string]: "252, 208, 142",
            } as React.CSSProperties
          }
        >
          <div className="img"></div>
        </div>
        <div
          className="card"
          style={
            {
              ["--index" as string]: 6,
              ["--color-card" as string]: "252, 142, 142",
            } as React.CSSProperties
          }
        >
          <div className="img"></div>
        </div>
        <div
          className="card"
          style={
            {
              ["--index" as string]: 7,
              ["--color-card" as string]: "252, 142, 239",
            } as React.CSSProperties
          }
        >
          <div className="img"></div>
        </div>
        <div
          className="card"
          style={
            {
              ["--index" as string]: 8,
              ["--color-card" as string]: "204, 142, 252",
            } as React.CSSProperties
          }
        >
          <div className="img"></div>
        </div>
        <div
          className="card"
          style={
            {
              ["--index" as string]: 9,
              ["--color-card" as string]: "142, 202, 252",
            } as React.CSSProperties
          }
        >
          <div className="img"></div>
        </div>
      </div>
    </div>
  );
};

export default RotatingCards;
