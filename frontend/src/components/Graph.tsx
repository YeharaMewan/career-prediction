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
    // Small delay to ensure the transition triggers after mount
    const timer = setTimeout(() => {
      setProgress(value);
    }, 100);

    return () => clearTimeout(timer);
  }, [value]);

  const offset = circumference - (progress / 100) * circumference;

  return (
    <div className="relative flex flex-col items-center justify-center">
      <svg
        width={size}
        height={size}
        style={{ transform: "rotate(-90deg)" }}
      >
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
          style={{ transition: "stroke-dashoffset 1.5s ease-out" }}
        />
      </svg>

      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <p className="text-3xl font-bold font-mono text-gray-800 leading-none">{value}%</p>
        {label && (
          <p className="text-lg font-semibold font-mono text-gray-500 uppercase tracking-wide -mt-1">
            {label}
          </p>
        )}
      </div>
    </div>
  );
};

export default Graph;
