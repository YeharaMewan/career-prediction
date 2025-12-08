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

// Skills Development Types
export interface SkillItem {
  name: string;
  description?: string;
  url?: string;
  platform?: string;
  cost?: string;
  duration?: string;
  provider?: string;
  difficulty?: string;
  rating?: string;
  enrolled_students?: string;
  learning_outcomes?: string[];
  resource_type?: string;
}

export interface SkillGroup {
  id: number;
  category: string;
  items: (string | SkillItem)[];
}

export interface SkillDevelopment {
  title: string;
  pathwayTitle: string;
  description: string;
  skillGroups: SkillGroup[];
}
