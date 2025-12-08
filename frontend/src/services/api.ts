/**
 * API Service for Career Planning Backend
 *
 * Provides methods to interact with the backend API endpoints
 */

const BASE_URL = 'http://localhost:8000';

/**
 * Get structured career pathway data for a session
 *
 * @param sessionId - The session ID from the career planning session
 * @returns Promise containing academic pathway and skill development data
 */
export const getCareerPathway = async (sessionId: string) => {
  const response = await fetch(`${BASE_URL}/api/career-pathway/${sessionId}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch career pathway: ${response.statusText}`);
  }

  return response.json();
};

/**
 * Initialize a new career planning session
 *
 * @param language - Preferred language ('en' or 'si')
 * @returns Promise containing session_id and first question
 */
export const initializeSession = async (language: string = 'en') => {
  const response = await fetch(`${BASE_URL}/session/initialize`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ language }),
  });

  if (!response.ok) {
    throw new Error(`Failed to initialize session: ${response.statusText}`);
  }

  return response.json();
};

/**
 * Send user response to the backend
 *
 * @param sessionId - The session ID
 * @param response - User's response text
 * @param language - Preferred language ('en' or 'si')
 * @returns Promise containing next question or career predictions
 */
export const sendResponse = async (sessionId: string, response: string, language: string = 'en') => {
  const res = await fetch(`${BASE_URL}/sessions/${sessionId}/respond`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ response, language }),
  });

  if (!res.ok) {
    throw new Error(`Failed to send response: ${res.statusText}`);
  }

  return res.json();
};

/**
 * Select a career and trigger planning agents
 *
 * @param sessionId - The session ID from Q&A
 * @param careerTitle - Selected career title
 * @param language - Preferred language ('en' or 'si')
 * @returns Promise containing planning status
 */
export const selectCareer = async (
  sessionId: string,
  careerTitle: string,
  language: string = 'en'
) => {
  const response = await fetch(`${BASE_URL}/sessions/${sessionId}/respond`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      response: `I want to pursue ${careerTitle}`,
      language,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to select career: ${response.statusText}`);
  }

  return response.json();
};

// Export all API methods as a single object
export default {
  getCareerPathway,
  initializeSession,
  sendResponse,
  selectCareer,
};
