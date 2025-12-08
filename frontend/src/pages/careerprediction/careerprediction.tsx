import { useState, useEffect, Component, ReactNode } from 'react';
import { BookOpen, Zap, CheckCircle, ExternalLink } from 'lucide-react';
import { useLocation } from 'react-router-dom';
import Footer from '../../components/Footer';
import apiService from '../../services/api';
import type { SkillItem } from '../../types/career';

// Error Boundary for catching render errors
class ErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('CareerPathway Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gradient-to-br from-cyan-50 via-teal-50 to-green-50 flex items-center justify-center px-4">
          <div className="bg-white rounded-2xl shadow-xl p-8 max-w-2xl">
            <h2 className="text-2xl font-bold text-red-600 mb-4">Error Loading Career Pathway</h2>
            <p className="text-gray-700 mb-4">
              There was an error rendering this page.
            </p>
            <pre className="bg-gray-100 p-4 rounded text-sm overflow-auto">
              {this.state.error?.message}
            </pre>
            <button
              onClick={() => window.location.reload()}
              className="mt-4 px-6 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700"
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Enhanced Course interface with detailed fields
interface Course {
  id: number;
  name: string;  // Institution name
  program_name?: string;  // Program/degree name
  duration: string;
  entry_requirements?: string[];  // Only for Professional Institutes
  cost_per_year?: string;
  total_cost?: string;
  scholarships?: string[];
  international_partnerships?: string[];
  website_url?: string;
  additional_notes?: string;
  details?: string;  // Legacy field for backward compatibility
}

// Subsection grouping cards (e.g., Government Universities, United Kingdom)
interface AcademicPathwaySubsection {
  subsectionTitle: string;
  cards: Course[];
}

// Main section (e.g., Local Pathways, International Pathway)
interface AcademicPathwaySection {
  sectionTitle: string;
  subsections: AcademicPathwaySubsection[];
}

// Complete academic pathway structure
interface AcademicData {
  title: string;
  pathwayTitle: string;
  description: string;
  courses?: Course[];  // Legacy support
  sections?: AcademicPathwaySection[];  // New hierarchical structure
}

// Skill group with category
interface SkillGroup {
  id: number;
  category: string;
  items: (string | SkillItem)[];
}

// Complete skills structure
interface SkillsData {
  title: string;
  pathwayTitle: string;
  description: string;
  skillGroups: SkillGroup[];
}

const ACADEMIC_DATA = {
  title: "Software Engineer",
  pathwayTitle: "Academic Pathway",
  description: "Build a strong foundation in computer science and software development",
  courses: [
    {
      id: 1,
      name: "Bachelor's Degree in Computer Science",
      duration: "4 years",
      details: "Core programming, algorithms, data structures, system design"
    },
    {
      id: 2,
      name: "Web Development Specialization",
      duration: "6 months",
      details: "Full-stack development, React, Node.js, databases"
    },
    {
      id: 3,
      name: "Advanced Algorithms & Data Structures",
      duration: "3 months",
      details: "Graph theory, dynamic programming, optimization techniques"
    },
    {
      id: 4,
      name: "Cloud Computing & DevOps",
      duration: "2 months",
      details: "AWS, Docker, Kubernetes, CI/CD pipelines"
    }
  ]
};

const SKILLS_DATA = {
  title: "Software Engineer",
  pathwayTitle: "Required Skills",
  description: "Develop these essential skills to excel in your career",
  skillGroups: [
    {
      id: 1,
      category: "Technical Skills",
      items: [
        "Programming Languages (Python, JavaScript, Java)",
        "Web Development (HTML, CSS, JavaScript)",
        "Backend Development (Node.js, Django, Spring)",
        "Database Design (SQL, MongoDB)",
        "Version Control (Git, GitHub)"
      ]
    },
    {
      id: 2,
      category: "Professional Skills",
      items: [
        "Problem Solving & Critical Thinking",
        "Team Collaboration & Communication",
        "Project Management & Agile Methodologies",
        "System Design & Architecture",
        "Documentation & Code Quality"
      ]
    },
    {
      id: 3,
      category: "Soft Skills",
      items: [
        "Adaptability & Continuous Learning",
        "Time Management",
        "Leadership & Mentorship",
        "Customer Focus & Empathy",
        "Creativity & Innovation"
      ]
    }
  ]
};

function CareerPathwayAppContent() {
  const [selectedPath, setSelectedPath] = useState<'academic' | 'skills'>('academic');
  const [academicData, setAcademicData] = useState<AcademicData>(ACADEMIC_DATA); // Fallback to static
  const [skillsData, setSkillsData] = useState<SkillsData>(SKILLS_DATA); // Fallback to static
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const location = useLocation();
  const sessionId = location.state?.sessionId;

  useEffect(() => {
    console.log('[CareerPathway] Component mounted', { sessionId });

    if (sessionId) {
      console.log('[CareerPathway] Fetching data for session:', sessionId);
      apiService.getCareerPathway(sessionId)
        .then(data => {
          console.log('[CareerPathway] API Response:', data);
          if (data.academic_pathway) {
            console.log('[CareerPathway] Setting academic data with',
              data.academic_pathway.sections?.length || 0, 'sections');
            setAcademicData(data.academic_pathway);
          }
          if (data.skill_development) {
            console.log('[CareerPathway] Setting skills data with',
              data.skill_development.skillGroups?.length || 0, 'groups');
            setSkillsData(data.skill_development);
          }
          setLoading(false);
        })
        .catch(err => {
          console.error('[CareerPathway] API Error:', err);
          setError(err.message);
          setLoading(false);
          // Fallback to static data on error
        });
    } else {
      console.log('[CareerPathway] No sessionId, using fallback data');
      setLoading(false);
    }
  }, [sessionId]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-cyan-50 via-teal-50 to-green-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-teal-600 mx-auto mb-4"></div>
          <p className="text-teal-800 text-lg">Loading your career pathway...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-cyan-50 via-teal-50 to-green-50 overflow-y-auto">
      <div className="min-h-[calc(100vh-100px)] px-4 md:px-12 py-12">
        <div className="max-w-4xl mx-auto">
          {/* Error Message */}
          {error && (
            <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 mb-6 rounded">
              <p className="font-bold">Note</p>
              <p>Unable to load dynamic data: {error}. Showing example data.</p>
            </div>
          )}

          {/* Page Title */}
          <h1 className="text-3xl md:text-4xl font-bold text-teal-800 text-center mb-8">
            {academicData.title}
          </h1>

          {/* Tab Button Group */}
          <TabButtonGroup
            selectedPath={selectedPath}
            onSelect={setSelectedPath}
          />

          {/* Content Card */}
          <div className="bg-white/80 backdrop-blur-md rounded-3xl shadow-xl p-8 md:p-12 border border-teal-100">
            <SectionHeader
              title={selectedPath === 'academic' ? academicData.pathwayTitle : skillsData.pathwayTitle}
              selectedPath={selectedPath}
              description={selectedPath === 'academic' ? academicData.description : skillsData.description}
            />

            {selectedPath === 'academic' ? (
              <AcademicPathwayComponent
                sections={academicData.sections}
                courses={academicData.courses}
              />
            ) : (
              <RequiredSkillsComponent skillGroups={skillsData.skillGroups} />
            )}
          </div>
        </div>
      </div>

      <Footer />
    </div>
  );
}

export default function CareerPathwayApp() {
  return (
    <ErrorBoundary>
      <CareerPathwayAppContent />
    </ErrorBoundary>
  );
}

interface TabButtonGroupProps {
  selectedPath: 'academic' | 'skills';
  onSelect: (path: 'academic' | 'skills') => void;
}

function TabButtonGroup({ selectedPath, onSelect }: TabButtonGroupProps) {
  return (
    <div className="flex justify-center mb-8">
      <div className="relative inline-flex rounded-full bg-white/80 p-1.5 backdrop-blur-sm ring-2 ring-teal-200 shadow-lg">
        {/* Academic Pathway Tab */}
        <button
          onClick={() => onSelect('academic')}
          className={`relative z-10 rounded-full px-6 py-3 text-sm font-semibold transition-all duration-300 flex items-center space-x-2 ${selectedPath === 'academic'
            ? 'text-white bg-gradient-to-r from-teal-500 to-cyan-600 shadow-md'
            : 'text-gray-600 hover:text-gray-900'
            }`}
        >
          <BookOpen size={18} />
          <span>Academic Pathway</span>
        </button>

        {/* Required Skills Tab */}
        <button
          onClick={() => onSelect('skills')}
          className={`relative z-10 rounded-full px-6 py-3 text-sm font-semibold transition-all duration-300 flex items-center space-x-2 ${selectedPath === 'skills'
            ? 'text-white bg-gradient-to-r from-cyan-500 to-teal-600 shadow-md'
            : 'text-gray-600 hover:text-gray-900'
            }`}
        >
          <Zap size={18} />
          <span>Required Skills</span>
        </button>
      </div>
    </div>
  );
}

interface SectionHeaderProps {
  title: string;
  selectedPath: 'academic' | 'skills';
  description: string;
}

function SectionHeader({ title, selectedPath, description }: SectionHeaderProps) {
  return (
    <div className="mb-10">
      <div className="flex items-center space-x-3 mb-6">
        {selectedPath === 'academic' ? (
          <BookOpen className="text-teal-500" size={32} />
        ) : (
          <Zap className="text-cyan-500" size={32} />
        )}
        <h2 className="text-3xl font-bold text-gray-900">{title}</h2>
      </div>
      <p className="text-gray-600 text-lg">{description}</p>
    </div>
  );
}

interface AcademicPathwayComponentProps {
  sections?: AcademicPathwaySection[];
  courses?: Course[];
}

function AcademicPathwayComponent({ sections, courses }: AcademicPathwayComponentProps) {
  // Use hierarchical sections if available (from backend), otherwise use courses (static fallback)
  if (sections && sections.length > 0) {
    return (
      <div className="space-y-8">
        {sections.map((section, sectionIdx) => (
          <div key={sectionIdx} className="space-y-6">
            {/* Section Title (e.g., "Local Pathways", "International Pathway") */}
            <h3 className="text-2xl font-bold text-gray-900 border-b-2 border-teal-200 pb-2">
              {section.sectionTitle}
            </h3>
            {/* Subsections (e.g., "Government Universities", "United Kingdom") */}
            {section.subsections?.map((subsection, subIdx) => (
              <div key={subIdx} className="space-y-4">
                <h4 className="text-xl font-semibold text-teal-700">
                  {subsection.subsectionTitle}
                </h4>
                {/* Cards */}
                <div className="space-y-3">
                  {subsection.cards?.map((card, cardIdx) => (
                    <CourseCard
                      key={card.id}
                      course={card}
                      index={cardIdx}
                      total={subsection.cards.length}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    );
  }

  // Fallback to original courses display (for backward compatibility)
  return (
    <div className="space-y-4">
      <h3 className="text-xl font-bold text-gray-900 mb-6">Recommended Courses & Programs</h3>
      {courses?.map((course, idx) => (
        <CourseCard key={course.id} course={course} index={idx} total={courses.length} />
      ))}
    </div>
  );
}

interface CourseCardProps {
  course: Course;
  index: number;
  total: number;
}

function CourseCard({ course, index, total }: CourseCardProps) {
  return (
    <div className="border-2 border-teal-200 rounded-2xl p-6 hover:shadow-lg transition-all duration-300 hover:border-teal-400">
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h4 className="text-lg font-bold text-gray-900 flex items-center space-x-2">
            <CheckCircle size={20} className="text-teal-500 flex-shrink-0" />
            <span>{course.name}</span>
          </h4>
          {course.program_name && (
            <p className="text-md text-gray-700 font-medium mt-1">{course.program_name}</p>
          )}
          <p className="text-sm text-teal-600 font-semibold mt-1">Duration: {course.duration}</p>
        </div>
        <span className="px-4 py-2 bg-teal-100 text-teal-700 rounded-full text-sm font-semibold flex-shrink-0">
          {index + 1}/{total}
        </span>
      </div>

      {/* Entry Requirements (only for Professional Institutes) */}
      {course.entry_requirements && course.entry_requirements.length > 0 && (
        <div className="mt-3">
          <p className="text-sm font-semibold text-gray-700 mb-1">Entry Requirements:</p>
          <ul className="list-disc list-inside text-gray-600 text-sm space-y-1">
            {course.entry_requirements.map((req, idx) => (
              <li key={idx}>{req}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Cost Information */}
      {course.cost_per_year && (
        <div className="mt-3">
          <p className="text-sm font-semibold text-gray-700">
            Cost: <span className="font-normal">{course.cost_per_year}</span>
          </p>
          {course.total_cost && (
            <p className="text-sm text-gray-600">Total: {course.total_cost}</p>
          )}
        </div>
      )}

      {/* Scholarships */}
      {course.scholarships && course.scholarships.length > 0 && (
        <div className="mt-3">
          <p className="text-sm font-semibold text-gray-700 mb-1">Scholarships:</p>
          <ul className="list-disc list-inside text-gray-600 text-sm space-y-1">
            {course.scholarships.map((scholarship, idx) => (
              <li key={idx}>{scholarship}</li>
            ))}
          </ul>
        </div>
      )}

      {/* International Partnerships */}
      {course.international_partnerships && course.international_partnerships.length > 0 && (
        <div className="mt-3">
          <p className="text-sm font-semibold text-gray-700 mb-1">International Partnerships:</p>
          <p className="text-gray-600 text-sm">{course.international_partnerships.join(', ')}</p>
        </div>
      )}

      {/* Website Link */}
      {course.website_url && (
        <div className="mt-3">
          <a
            href={course.website_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-teal-600 hover:text-teal-700 underline flex items-center space-x-1"
          >
            <span>Visit Website</span>
            <ExternalLink size={14} />
          </a>
        </div>
      )}

      {/* Additional Notes or Legacy Details (excluding Website URLs) */}
      {(course.additional_notes || course.details) && (
        <div className="mt-3 text-sm text-gray-600 italic">
          {(() => {
            const notes = course.additional_notes || course.details || '';
            // Remove "Website: <url>" pattern from notes
            const cleanedNotes = notes.replace(/\s*Website:\s*https?:\/\/[^\s]+/gi, '').trim();
            return cleanedNotes || null;
          })()}
        </div>
      )}
    </div>
  );
}

interface RequiredSkillsComponentProps {
  skillGroups: SkillGroup[];
}

function RequiredSkillsComponent({ skillGroups }: RequiredSkillsComponentProps) {
  // Group skill groups by main category (e.g., "Technical Skills", "Soft Skills", "Learning Road Map")
  const groupedCategories: { [key: string]: SkillGroup[] } = {};

  skillGroups.forEach(group => {
    // Extract main category (e.g., "Technical Skills" from "Technical Skills - Core Skills")
    const mainCategory = group.category.includes(' - ')
      ? group.category.split(' - ')[0]
      : group.category;

    if (!groupedCategories[mainCategory]) {
      groupedCategories[mainCategory] = [];
    }
    groupedCategories[mainCategory].push(group);
  });

  return (
    <div className="space-y-8">
      {Object.entries(groupedCategories).map(([mainCategory, groups]) => {
        // Determine if this is a simple category (Technical/Soft Skills) or comprehensive (Learning Roadmap, etc.)
        const isSimpleCategory = mainCategory === "Technical Skills" || mainCategory === "Soft Skills";

        return (
          <div key={mainCategory} className="space-y-6">
            {/* Main Category Title (NOT a card, similar to Academic Pathway section titles) */}
            <h3 className="text-2xl font-bold text-gray-900 border-b-2 border-cyan-200 pb-2">
              {mainCategory}
            </h3>

            {/* Subcategory Cards or Direct Items */}
            <div className="space-y-4">
              {groups.map(skillGroup => {
                // Extract subcategory title (e.g., "Core Skills" from "Technical Skills - Core Skills")
                const subcategoryTitle = skillGroup.category.includes(' - ')
                  ? skillGroup.category.split(' - ').slice(1).join(' - ')
                  : null;

                // For simple categories (Technical/Soft Skills), show subcategory cards with simple items
                if (isSimpleCategory) {
                  return (
                    <div key={skillGroup.id} className="border-2 border-cyan-200 rounded-2xl p-6 hover:shadow-lg transition-all duration-300 hover:border-cyan-400">
                      {/* Subcategory Title inside card */}
                      {subcategoryTitle && (
                        <h4 className="text-xl font-semibold text-cyan-700 mb-4">
                          {subcategoryTitle}
                        </h4>
                      )}
                      {/* Simple string items in grid */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {skillGroup.items.map((skill, i) => (
                          <SkillItem key={i} skill={skill} />
                        ))}
                      </div>
                    </div>
                  );
                }

                // For comprehensive categories (Learning Roadmap, Certifications, Resources),
                // show subcategory title and comprehensive cards
                return (
                  <div key={skillGroup.id} className="space-y-4">
                    {/* Subcategory Title (e.g., "Foundation (0-6 months)", "Practice Platforms") */}
                    {subcategoryTitle && (
                      <h4 className="text-xl font-semibold text-cyan-700">
                        {subcategoryTitle}
                      </h4>
                    )}
                    {/* Comprehensive academic-style cards */}
                    <div className="space-y-3">
                      {skillGroup.items.map((skill, i) => (
                        <SkillItem key={i} skill={skill} />
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

interface SkillItemProps {
  skill: string | SkillItem;
}

function SkillItem({ skill }: SkillItemProps) {
  // Simple string skill (for Technical/Soft Skills - KEEP AS IS)
  if (typeof skill === 'string') {
    return (
      <div className="flex items-start space-x-3 bg-cyan-50 rounded-lg p-3">
        <CheckCircle size={18} className="text-cyan-500 flex-shrink-0 mt-0.5" />
        <span className="text-gray-700">{skill}</span>
      </div>
    );
  }

  // Enhanced SkillItem with full details (ACADEMIC CARD STYLE for Learning Roadmap, Certifications, Resources)
  return (
    <div className="border-2 border-cyan-200 rounded-2xl p-6 hover:shadow-lg transition-all duration-300 hover:border-cyan-400 bg-white">
      {/* Header with name */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h4 className="text-lg font-bold text-gray-900 flex items-center space-x-2">
            <CheckCircle size={20} className="text-cyan-500 flex-shrink-0" />
            <span>{skill.name}</span>
          </h4>
          {skill.description && (
            <p className="text-md text-gray-700 mt-2">{skill.description}</p>
          )}
        </div>
      </div>

      {/* Platform & Provider */}
      <div className="mt-3 space-y-2">
        {skill.platform && (
          <p className="text-sm text-gray-700">
            <span className="font-semibold">Platform:</span> {skill.platform}
          </p>
        )}
        {skill.provider && (
          <p className="text-sm text-gray-700">
            <span className="font-semibold">Provider:</span> {skill.provider}
          </p>
        )}
      </div>

      {/* Cost, Duration, Difficulty badges */}
      <div className="mt-3 flex flex-wrap gap-2">
        {skill.cost && (
          <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-semibold">
            {skill.cost}
          </span>
        )}
        {skill.duration && (
          <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-semibold">
            {skill.duration}
          </span>
        )}
        {skill.difficulty && (
          <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm font-semibold">
            {skill.difficulty}
          </span>
        )}
      </div>

      {/* Rating & Enrolled Students */}
      {(skill.rating || skill.enrolled_students) && (
        <div className="mt-3 flex gap-4 text-sm text-gray-600">
          {skill.rating && (
            <span className="flex items-center gap-1">
              <span>⭐</span> {skill.rating}
            </span>
          )}
          {skill.enrolled_students && (
            <span className="flex items-center gap-1">
              <span>👥</span> {skill.enrolled_students}
            </span>
          )}
        </div>
      )}

      {/* Learning Outcomes */}
      {skill.learning_outcomes && skill.learning_outcomes.length > 0 && (
        <div className="mt-4 bg-cyan-50 rounded-lg p-4">
          <p className="text-sm font-semibold text-gray-800 mb-2">Learning Outcomes:</p>
          <ul className="list-disc list-inside text-gray-700 text-sm space-y-1">
            {skill.learning_outcomes.map((outcome, idx) => (
              <li key={idx}>{outcome}</li>
            ))}
          </ul>
        </div>
      )}

      {/* URL Link */}
      {skill.url && (
        <div className="mt-4">
          <a
            href={skill.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-cyan-600 hover:text-cyan-700 font-semibold underline flex items-center space-x-2"
          >
            <span>Visit Course/Resource →</span>
            <ExternalLink size={14} />
          </a>
        </div>
      )}
    </div>
  );
}