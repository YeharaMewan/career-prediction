export interface CareerPrediction {
  title: string;
  description: string;
  confidence_score: number; // 0.0 to 1.0 (convert to percentage for display)
  required_skills?: string[];
  median_salary?: string;
  why_good_fit?: string;
  growth_outlook?: string;
  riasec_alignment?: string;
  career_path?: string;
}
