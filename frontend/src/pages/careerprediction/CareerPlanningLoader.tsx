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
    console.error('[Loader] Missing required data:', { sessionId, careerTitle });
    navigate('/chat');
    return;
  }

  console.log('[Loader] Starting career planning flow:', { sessionId, careerTitle, language });

  const triggerAndPoll = async () => {
    try {
      // Trigger career planning
      console.log('[Loader] Triggering career planning via API...');
      await apiService.selectCareer(sessionId, careerTitle, language);
      console.log('[Loader] Career planning triggered successfully');

      // Wait 90 seconds for agents to complete (they run in parallel and take 60-90s)
      console.log('[Loader] Waiting 90 seconds for agents to process...');
      await new Promise(resolve => setTimeout(resolve, 90000));

      // Now poll for results
      console.log('[Loader] Starting to poll for results...');
      await pollForResults();

    } catch (err: any) {
      console.error('[Loader] Error in flow:', err);
      setError(err.message || 'Failed to process career planning');
      
      // Still navigate after error
      setTimeout(() => {
        navigate('/career-pathway', { state: { sessionId } });
      }, 3000);
    }
  };

  const pollForResults = async () => {
    const maxAttempts = 5; // Only 5 attempts since we already waited 90s
    let attempts = 0;

    const poll = async (): Promise<void> => {
      attempts++;
      console.log(`[Loader] Poll attempt ${attempts}/${maxAttempts}`);

      try {
        const data = await apiService.getCareerPathway(sessionId);
        
        if (data.academic_pathway?.sections?.length > 0 || data.skill_development?.skillGroups?.length > 0) {
          console.log('[Loader] ✅ Data ready! Navigating...');
          navigate('/career-pathway', { state: { sessionId } });
          return;
        }
        
        console.log('[Loader] Data not ready yet, retrying...');
      } catch (err: any) {
        console.log(`[Loader] Poll attempt ${attempts} failed:`, err.message);
      }

      if (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 3000)); // Wait 3s between polls
        return poll();
      } else {
        console.log('[Loader] Max attempts reached, navigating anyway...');
        navigate('/career-pathway', { state: { sessionId } });
      }
    };

    return poll();
  };

  triggerAndPoll();
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
