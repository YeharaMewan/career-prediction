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
          className="font-space-grotesk text-[40px] font-bold text-white"
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
              ["--color-card" as string]: "34, 87, 122",
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
              ["--color-card" as string]: "56, 163, 165",
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
              ["--color-card" as string]: "87, 204, 153",
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
              ["--color-card" as string]: "128, 237, 153",
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
              ["--color-card" as string]: "199, 249, 204",
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
              ["--color-card" as string]: "27, 70, 98",
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
              ["--color-card" as string]: "70, 204, 207",
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
              ["--color-card" as string]: "78, 184, 138",
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
              ["--color-card" as string]: "153, 255, 178",
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
              ["--color-card" as string]: "220, 255, 225",
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
