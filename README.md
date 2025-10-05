# Exoplanet Predictor

A Next.js application that uses Google's Gemini LLM to analyze exoplanet data from the Kepler dataset and generate AI-powered visualizations.

## Features

- **AI-Powered Analysis**: Uses Gemini 2.5 Pro to analyze Kepler dataset parameters and determine if an object is an exoplanet
- **Two-Column Layout**: Displays AI analysis alongside AI-generated exoplanet visualizations
- **Kepler Dataset Integration**: Processes real NASA Kepler mission data including orbital periods, transit properties, and stellar characteristics
- **Interactive Forms**: User-friendly interface for inputting exoplanet parameters
- **Real-time Results**: Instant analysis and visualization generation
- **Responsive Design**: Works on desktop and mobile devices
- **Docker Support**: Easy deployment with Docker and Docker Compose

## Technology Stack

- **Frontend**: Next.js 15, React 19, TypeScript
- **Styling**: Tailwind CSS with custom dark theme
- **AI Integration**: Google Gemini 2.5 Pro LLM
- **UI Components**: Radix UI components
- **Deployment**: Docker, Docker Compose

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or pnpm
- Google Gemini API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/riniann9/NASA_SA_Test.git
cd NASA_SA_Test
```

2. Install dependencies:
```bash
npm install
# or
pnpm install
```

3. Set up environment variables:
```bash
# Create .env.local file
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env.local
```

4. Run the development server:
```bash
npm run dev
# or
pnpm dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

### Docker Deployment

1. Build and run with Docker Compose:
```bash
docker-compose up -d
```

2. Or use the deployment script:
```bash
chmod +x deploy.sh
./deploy.sh
```

## Usage

1. **Navigate to `/new`** to input exoplanet parameters
2. **Fill out the form** with Kepler dataset values:
   - Orbital & Transit Properties
   - Planetary Properties  
   - Detection & Signal Properties
   - Stellar Properties
   - Observational Properties
3. **Submit for analysis** - the Gemini LLM will analyze your data
4. **View results** on the results page with:
   - AI Analysis with confidence scores
   - AI Generated Exoplanet visualization
   - Key Features Impact analysis

## API Endpoints

- `POST /api/analyze-planet` - Analyzes exoplanet data using Gemini LLM
- `POST /api/generate-planet-image` - Generates AI exoplanet visualizations

## Project Structure

```
exoplanet-predictor/
├── app/
│   ├── api/
│   │   ├── analyze-planet/
│   │   └── generate-planet-image/
│   ├── new/
│   ├── results/
│   └── page.tsx
├── components/
│   ├── ui/
│   └── ...
├── lib/
├── public/
└── styles/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- NASA Kepler Mission data
- Google Gemini LLM
- Next.js and React communities
