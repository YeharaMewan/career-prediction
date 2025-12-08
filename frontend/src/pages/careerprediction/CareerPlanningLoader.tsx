import { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import CareerLoadingComponent from '../../components/Loading';
import apiService from '../../services/api';

interface LocationState {
  sessionId: string;
  careerTitle: string;
  language?: string;
}

export default function CareerPlanningLoader() {
  const navigate = useNavigate();
  const location = useLocation();
  const [error, setError] = useState<string | null>(null);

  const { sessionId, careerTitle, language = 'en' } = (location.state || {}) as LocationState;

  useEffect(() => {
    if (!sessionId || !careerTitle) {
      // Redirect to chat if missing required data
      navigate('/chat');
      return;
    }

    // Trigger career planning
    apiService.selectCareer(sessionId, careerTitle, language)
      .then(() => {
        // Career planning initiated successfully
        // Wait a moment for agents to complete (they run in parallel ~60s)
        // Poll or wait for completion

        // For now, navigate after a delay (you can improve this with polling)
        setTimeout(() => {
          navigate('/career-pathway', {
            state: { sessionId },
          });
        }, 65000); // 65 seconds (60s for agents + 5s buffer)
      })
      .catch((err) => {
        console.error('Failed to trigger career planning:', err);
        setError(err.message);
        // Navigate anyway after showing error briefly
        setTimeout(() => {
          navigate('/career-pathway', {
            state: { sessionId },
          });
        }, 3000);
      });
  }, [sessionId, careerTitle, language, navigate]);

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-cyan-50 via-teal-50 to-green-50 flex items-center justify-center">
        <div className="text-center bg-white p-8 rounded-lg shadow-lg">
          <p className="text-red-600 mb-4">⚠️ {error}</p>
          <p className="text-gray-600">Redirecting to career pathway...</p>
        </div>
      </div>
    );
  }

  return <CareerLoadingComponent />;
}
