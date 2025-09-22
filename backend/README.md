# Career Planning LLM System

A comprehensive hierarchical multi-agent system for personalized career planning, built with LangGraph and powered by OpenAI's GPT models.

## Architecture Overview

This system implements a hierarchical supervisor architecture with specialized agents:

### **Main Supervisor (CEO)**
- Orchestrates the entire career planning workflow
- Coordinates between all specialized teams
- Generates comprehensive final reports

### **Career Planning Supervisor (Team Lead)**
- **Dual Role**: Career exploration + Team orchestration
- Identifies suitable careers based on student profiles
- Manages Academic and Skill Development agents
- Creates complete career blueprints

### **Worker Agents**
- **User Profiler Agent**: Creates comprehensive student profiles
- **Academic Pathway Agent**: Designs education plans and degree pathways  
- **Skill Development Agent**: Creates skill development roadmaps
- **Future Trends Analyst**: Analyzes career viability and market trends

## Features

- **Comprehensive Student Profiling**: Analyzes academic background, skills, interests, and goals
- **Multiple Career Options**: Identifies and analyzes 3 suitable career paths
- **Academic Planning**: Detailed degree requirements and educational pathways
- **Skill Development**: Progressive skill building plans with timelines
- **Market Analysis**: Future trends, automation impact, and viability scoring
- **Web Research Integration**: Real-time market data via Tavily search
- **Detailed Reporting**: Professional career planning reports in Markdown

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd backend
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Setup**
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required API keys:
- `OPENAI_API_KEY`: OpenAI API key for LLM access
- `TAVILY_API_KEY`: Tavily API key for web search

## Usage

### Interactive Mode
```bash
python main.py --interactive
```

### Direct Query
```bash
python main.py "I am a computer science student interested in AI and machine learning"
```

### Test Mode
```bash
python main.py --test
```

## System Workflow

1. **User Profiling**: Comprehensive analysis of student background, skills, interests, and goals
2. **Career Exploration**: AI-powered identification of 3 suitable career paths
3. **Blueprint Creation**: For each career:
   - Academic pathway planning (degrees, certifications, timeline)
   - Skill development roadmap (technical and soft skills)
   - Career-specific information (responsibilities, progression, salary)
4. **Market Analysis**: Future viability assessment using real-time web research
5. **Report Generation**: Comprehensive career planning report with recommendations

## File Structure

```
backend/
├── main.py                          # Main entry point
├── requirements.txt                 # Dependencies
├── .env                            # Environment variables
├── README.md                       # This file
│
├── config/
│   ├── settings.py                 # Configuration settings
│   └── prompts.py                  # Agent prompts
│
├── agents/
│   ├── base_agent.py               # Base agent classes
│   ├── supervisors/
│   │   ├── main_supervisor.py      # Main CEO supervisor
│   │   └── career_planning_supervisor.py  # Career team supervisor
│   └── workers/
│       ├── user_profiler.py        # User profiling agent
│       ├── academic_pathway.py     # Academic planning agent
│       ├── skill_development.py    # Skill development agent
│       └── future_trends.py        # Future trends analyst
│
├── tools/
│   └── web_search.py              # Web search functionality
│
├── models/
│   └── state_models.py            # Data models
│
├── utils/
│   └── handoff_tools.py           # Agent coordination tools
│
└── outputs/
    ├── reports/                   # Generated reports
    └── logs/                      # System logs
```

## Configuration

### Environment Variables

All configuration is handled through environment variables. Copy `.env.example` to `.env` and configure:

- **OpenAI Settings**: API key, model selection, temperature
- **Search Integration**: Tavily API for real-time research
- **System Settings**: Logging, timeouts, output paths
- **Agent Behavior**: Parallel processing limits, iteration counts

### Agent Prompts

All agent prompts are centralized in `config/prompts.py` for easy customization and maintenance.

## Agent Specifications

### Main Supervisor
- **Role**: CEO/Executive coordination
- **Responsibilities**: High-level workflow management, final report generation
- **Tools**: Agent handoff tools for delegation

### Career Planning Supervisor  
- **Role**: Team lead with dual responsibilities
- **Phase 1**: Career exploration and matching
- **Phase 2**: Team orchestration for blueprint creation
- **Tools**: Academic and skill development agent coordination

### User Profiler Agent
- **Specialization**: Student profile analysis
- **Output**: Comprehensive StudentProfile object
- **Analysis**: Academic background, skills, interests, goals, constraints

### Academic Pathway Agent
- **Specialization**: Education planning
- **Research**: Web search for degree requirements and certifications
- **Output**: Detailed academic timeline and pathways

### Skill Development Agent
- **Specialization**: Skill roadmap creation
- **Analysis**: Technical skills, soft skills, learning resources
- **Output**: Progressive skill development plan with milestones

### Future Trends Analyst
- **Specialization**: Market analysis and career viability
- **Research**: Industry trends, automation impact, job market data
- **Output**: Viability scoring with risk assessment

## Output Examples

### Career Report Structure
```
# COMPREHENSIVE CAREER PLANNING REPORT
├── Executive Summary
├── Student Profile Summary  
├── Career Recommendations (3 careers)
│   ├── Career Overview & Match Score
│   ├── Future Viability Analysis
│   ├── Academic Pathway Summary
│   ├── Skill Development Plan
│   └── Salary Expectations
├── Market Analysis & Future Outlook
├── Recommended Action Plan
└── Next Steps
```

### Career Blueprint Components
- **Academic Plan**: Degrees, certifications, timeline, institutions
- **Skill Plan**: Technical skills, soft skills, learning resources, milestones
- **Career Info**: Responsibilities, progression, salary ranges
- **Match Analysis**: Student compatibility scoring and reasoning

## Development

### Adding New Agents
1. Create agent class inheriting from `BaseAgent` or `WorkerAgent`
2. Define specialized tools and prompts
3. Implement `process_task()` method
4. Add handoff tools for coordination
5. Register in the multi-agent graph

### Customizing Analysis
- **Career Matching**: Modify logic in `CareerPlanningSupervisor._identify_suitable_careers()`
- **Skills Assessment**: Update keyword mapping in skill development agent
- **Market Analysis**: Enhance web search queries and analysis methods
- **Report Format**: Customize `MainSupervisor._generate_final_report()`

## API Integration

### Current Integrations
- **OpenAI gpt-4-mini**: Core reasoning and analysis
- **Tavily Search**: Real-time web research for market data
- **LangSmith**: Optional monitoring and debugging

### Adding New Tools
Extend the tools directory with new search engines, databases, or analysis services following the existing pattern.

## Monitoring & Debugging

- **Logging**: Comprehensive logging to files and console
- **LangSmith**: Optional integration for trace monitoring
- **Session Tracking**: Unique session IDs for request tracing
- **Error Handling**: Graceful degradation with detailed error messages

## Performance Considerations

- **Sequential Processing**: Agents process sequentially for reliability
- **Timeout Management**: Configurable timeouts for web requests
- **Memory Management**: Efficient state handling between agents
- **Report Caching**: Generated reports saved to filesystem

## Security & Privacy

- **API Key Management**: Environment variable configuration
- **Data Handling**: No persistent storage of user data
- **Session Isolation**: Each request processed independently
- **Input Validation**: Sanitization of user inputs

## Troubleshooting

### Common Issues
1. **Missing API Keys**: Ensure `.env` file is properly configured
2. **Network Timeouts**: Check internet connection and API limits
3. **Agent Failures**: Review logs for specific error messages
4. **Report Generation**: Verify output directory permissions

### Debug Mode
```bash
DEBUG=true LOG_LEVEL=DEBUG python main.py --interactive
```

## Contributing

1. Follow the established architecture patterns
2. Add comprehensive logging to new components
3. Include error handling and graceful degradation
4. Update documentation for new features
5. Test with various input scenarios

## License

[Add your license information here]

---

For questions or support, please review the logs and documentation, or contact the development team.