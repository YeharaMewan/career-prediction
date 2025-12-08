import { useState, useEffect } from 'react';

interface StepState {
  status: 'pending' | 'active' | 'completed';
}

export default function CareerLoadingComponent() {
  const [isLoading, setIsLoading] = useState(true);
  const [steps, setSteps] = useState<StepState[]>([
    { status: 'active' },   // Step 1 starts immediately
    { status: 'pending' },  // Step 2 pending
    { status: 'pending' },  // Step 3 pending
  ]);
  const [timeRemaining, setTimeRemaining] = useState(90); // Countdown from 90s

  const simulateLoading = () => {
    setIsLoading(true);
    setTimeout(() => setIsLoading(false), 4000);
  };

  // Step transitions: 30s per step
  useEffect(() => {
    // Step 1 → Step 2 at 30s
    const timer1 = setTimeout(() => {
      setSteps([
        { status: 'completed' },
        { status: 'active' },
        { status: 'pending' },
      ]);
    }, 30000);

    // Step 2 → Step 3 at 60s
    const timer2 = setTimeout(() => {
      setSteps([
        { status: 'completed' },
        { status: 'completed' },
        { status: 'active' },
      ]);
    }, 60000);

    // Step 3 complete at 90s
    const timer3 = setTimeout(() => {
      setSteps([
        { status: 'completed' },
        { status: 'completed' },
        { status: 'completed' },
      ]);
    }, 90000);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
    };
  }, []);

  // Countdown timer
  useEffect(() => {
    const interval = setInterval(() => {
      setTimeRemaining((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Helper function to format time as M:SS
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Helper function to render step icon (checkmark, spinner, or dot)
  const renderStepIcon = (status: 'pending' | 'active' | 'completed') => {
    if (status === 'completed') {
      // Green checkmark
      return (
        <svg className="w-5 h-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
        </svg>
      );
    } else if (status === 'active') {
      // Spinning loader
      return (
        <div className="w-5 h-5 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
      );
    } else {
      // Pending (gray dot)
      return (
        <svg className="w-5 h-5 text-gray-300" fill="currentColor" viewBox="0 0 20 20">
          <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
        </svg>
      );
    }
  };

  // Helper function to render colored dot
  const renderStepDot = (status: 'pending' | 'active' | 'completed') => {
    if (status === 'completed') {
      return <div className="w-3 h-3 bg-green-500 rounded-full mr-4 flex-shrink-0"></div>;
    } else if (status === 'active') {
      return <div className="w-3 h-3 bg-cyan-400 rounded-full mr-4 flex-shrink-0 animate-pulse"></div>;
    } else {
      return <div className="w-3 h-3 bg-gray-300 rounded-full mr-4 flex-shrink-0"></div>;
    }
  };

  // Helper function to get text color based on status
  const getStepTextColor = (status: 'pending' | 'active' | 'completed') => {
    return status === 'pending' ? 'text-gray-500' : 'text-gray-700';
  };

  if (!isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-cyan-50 via-teal-50 to-green-50">
        <div className="text-center">
          <div className="text-teal-500 mb-4 animate-bounce">
            <svg className="w-20 h-20 mx-auto" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          </div>
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Prediction Ready!</h2>
          <p className="text-gray-600 mb-6">Your career insights have been generated</p>
          <button
            onClick={simulateLoading}
            className="px-8 py-3 bg-teal-500 text-white rounded-full hover:bg-teal-600 transition font-semibold"
          >
            Analyze Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-cyan-50 via-teal-50 to-green-50">
      <div className="w-full max-w-2xl px-4">

        <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl p-12 text-center border border-teal-100">

          <div className="mb-8 flex justify-center">
            <div className="relative w-32 h-32">

              <div className="absolute inset-0 border-2 border-cyan-300 rounded-full"></div>


              <svg className="absolute inset-0 w-full h-full animate-spin" style={{ animationDuration: '2s' }} viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" fill="none" stroke="url(#gradient)" strokeWidth="5" strokeDasharray="70 220" strokeLinecap="round" />
                <defs>
                  <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#14b8a6" />
                    <stop offset="100%" stopColor="#06b6d4" />
                  </linearGradient>
                </defs>
              </svg>


              <div className="absolute inset-0 flex items-center justify-center">
                <svg className="w-16 h-16 animate-bounce" viewBox="0 0 100 100" style={{ animationDuration: '1.5s' }}>

                  <circle cx="50" cy="40" r="20" fill="#14b8a6" />


                  <circle cx="43" cy="37" r="2.5" fill="white" />
                  <circle cx="57" cy="37" r="2.5" fill="white" />
                  <circle cx="42" cy="36" r="1.2" fill="#0f766e" />
                  <circle cx="56" cy="36" r="1.2" fill="#0f766e" />


                  <path d="M 43 43 Q 50 46 57 43" stroke="white" strokeWidth="1.5" fill="none" strokeLinecap="round" />


                  <rect x="40" y="60" width="20" height="18" rx="3" fill="#06b6d4" />


                  <rect x="28" y="65" width="10" height="6" rx="3" fill="#06b6d4" />
                  <rect x="62" y="65" width="10" height="6" rx="3" fill="#06b6d4" />


                  <rect x="43" y="78" width="5" height="10" rx="2" fill="#14b8a6" />
                  <rect x="52" y="78" width="5" height="10" rx="2" fill="#14b8a6" />
                </svg>
              </div>


            </div>
          </div>


          <h2 className="text-3xl font-bold text-gray-900 mb-2">
            Academic Pathways and Soft Skills
          </h2>
          <p className="text-gray-600 text-lg mb-10">
            Let AI analyze your profile and discover the perfect academic paths for you...
          </p>


          {/* Step Progress Section */}
          <div className="space-y-4 mb-10">
            {/* Step 1: Analyzing Academic Pathways */}
            <div className="flex items-center justify-start px-4">
              {renderStepDot(steps[0].status)}
              <div className="text-left flex-1">
                <span className={`font-medium ${getStepTextColor(steps[0].status)}`}>
                  Analyzing Your Academic Pathways
                </span>
              </div>
              {renderStepIcon(steps[0].status)}
            </div>

            {/* Step 2: Matching Skills & Experience */}
            <div className="flex items-center justify-start px-4">
              {renderStepDot(steps[1].status)}
              <div className="text-left flex-1">
                <span className={`font-medium ${getStepTextColor(steps[1].status)}`}>
                  Matching With Your Skills & Experience
                </span>
              </div>
              {renderStepIcon(steps[1].status)}
            </div>

            {/* Step 3: Generating Personalized Insights */}
            <div className="flex items-center justify-start px-4">
              {renderStepDot(steps[2].status)}
              <div className="text-left flex-1">
                <span className={`font-medium ${getStepTextColor(steps[2].status)}`}>
                  Generating Your Personalized Insights
                </span>
              </div>
              {renderStepIcon(steps[2].status)}
            </div>
          </div>


          <div className="flex items-center justify-center space-x-1 mb-2">
            <span className="text-gray-600 font-medium">Thinking</span>
            <div className="flex space-x-1">
              <span className="w-2.5 h-2.5 bg-teal-500 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></span>
              <span className="w-2.5 h-2.5 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0.15s' }}></span>
              <span className="w-2.5 h-2.5 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '0.3s' }}></span>
            </div>
          </div>


          <p className="text-sm text-gray-500">Estimated time: {formatTime(timeRemaining)}</p>
        </div>


        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white/60 backdrop-blur rounded-2xl p-4 text-center border border-teal-100">
            <p className="text-sm text-gray-600"><span className="font-semibold text-teal-600"> Instant </span> Responses</p>
          </div>
          <div className="bg-white/60 backdrop-blur rounded-2xl p-4 text-center border border-teal-100">
            <p className="text-sm text-gray-600"><span className="font-semibold text-teal-600"> Private </span> & Secure</p>
          </div>
          <div className="bg-white/60 backdrop-blur rounded-2xl p-4 text-center border border-teal-100">
            <p className="text-sm text-gray-600"><span className="font-semibold text-teal-600"> Creative</span> & Logical</p>
          </div>
        </div>
      </div>
    </div>
  );
}