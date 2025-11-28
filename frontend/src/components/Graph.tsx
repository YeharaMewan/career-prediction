import React, { useEffect, useState } from "react";

interface GraphProps {
  value: number;         
  size?: number;         
  strokeWidth?: number; 
  label?: string;   
  strokeColor?: string;     
}

const Graph: React.FC<GraphProps> = ({
  value,
  strokeColor = "#00A8FF",
  size = 120,
  strokeWidth = 10,
  label = "",
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;

  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let start = 0;
    const speed = 20; 

    const interval = setInterval(() => {
      start += 1;
      setProgress(start);

      if (start >= value) clearInterval(interval);
    }, speed);

    return () => clearInterval(interval);
  }, [value]);

  const offset = circumference - (progress / 100) * circumference;

  return (
    <div className="flex flex-col items-center justify-center">
      <svg width={size} height={size}>
        <circle
          stroke="#e5e7eb"
          fill="transparent"
          strokeWidth={strokeWidth}
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        <circle
          stroke={strokeColor}
          fill="transparent"
          strokeWidth={strokeWidth}
          r={radius}
          cx={size / 2}
          cy={size / 2}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 0.2s ease" }}
        />
      </svg>

      <div className="-mt-4 text-center absolute">
        <p className="text-3xl font-bold font-mono  text-gray-800">{progress}%</p>
        {label && (
          <p className="text-lg font-semibold font-mono  text-gray-500 mt-1 uppercase tracking-wide">
            {label}
          </p>
        )}
      </div>
    </div>
  );
};

export default Graph;
