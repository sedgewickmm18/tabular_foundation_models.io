# Foundation Models for Tables: TabPFN & TabICL

A comprehensive reveal.js presentation exploring foundation models for tabular data, with deep dives into TabPFN (Tabular Prior-Fitted Networks) and TabICL (Tabular In-Context Learning).

## 🌐 Live Demo

**[View Presentation](https://sedgewickmm18.github.io/tabular_foundation_models.io/)**

> Replace `[your-username]` and `[repo-name]` with your actual GitHub username and repository name.

## 📋 Overview

This presentation covers:
- **Use Cases**: Imputation and anomaly detection
- **Introduction**: Foundation models for tabular data
- **TabPFN Architecture**: Prior-fitted networks and transformer-based design
- **TabICL Architecture**: Improvements and scalability enhancements
- **Training Methodology**: Meta-learning and synthetic data generation
- **Code Examples**: Practical implementations with hyperparameter tuning
- **Fine-tuning**: Domain adaptation strategies
- **Benchmarks**: Performance comparisons
- **Limitations**: Current constraints and future directions

## 🚀 Quick Start

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone or download this repository**

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the presentation**
   ```bash
   npm start
   ```

   This will start a local server and open the presentation in your default browser at `http://localhost:8000`

## 📂 Project Structure

```
tabular/
├── index.html              # Main presentation file
├── package.json            # Dependencies and scripts
├── README.md              # This file
├── slides/                # Modular markdown slides
│   ├── 01-use-cases.md
│   ├── 02-introduction.md
│   ├── 03-tabpfn-architecture.md
│   ├── 04-tabicl-architecture.md
│   ├── 05-training-methodology.md
│   ├── 06-code-examples.md
│   ├── 07-fine-tuning.md
│   ├── 08-benchmarks.md
│   └── 09-limitations.md
├── assets/
│   └── css/
│       └── custom.css     # Custom styling
└── node_modules/          # Dependencies (generated)
    └── reveal.js/
```

## 🎮 Navigation

### Keyboard Shortcuts

- **→** or **Space**: Next slide (horizontal)
- **←**: Previous slide (horizontal)
- **↓**: Next vertical slide (deep dive)
- **↑**: Previous vertical slide
- **Home**: First slide
- **End**: Last slide
- **Esc** or **O**: Slide overview mode
- **S**: Speaker notes view
- **F**: Fullscreen mode
- **B** or **.**: Pause/blackout
- **?**: Show keyboard shortcuts help

### Navigation Structure

The presentation uses a 2D navigation model:
- **Horizontal (→)**: Move between main topics
- **Vertical (↓)**: Deep dive into current topic with detailed explanations, math, and code

```
Use Cases → Introduction → TabPFN → TabICL → Training → Code → Fine-tuning → Benchmarks → Limitations
    ↓            ↓           ↓         ↓         ↓        ↓         ↓            ↓            ↓
  Details      Papers    Math/Arch  Improvements Process Examples  Strategies   Results    Future
    ↓            ↓           ↓         ↓         ↓        ↓         ↓            ↓            ↓
  Examples    Evolution  Training   Scalability  Math    Chunking   Code      Analysis    Roadmap
```

## 🎯 Presentation Modes

### Quick Overview (30 minutes)
Navigate horizontally only through main slides for a high-level overview.

### Technical Deep Dive (60-90 minutes)
Include vertical slides for detailed technical content:
- Mathematical formulations
- Architecture details
- Code examples
- Performance analysis

### Workshop Format (2-3 hours)
Full vertical navigation with:
- All technical details
- Interactive discussions
- Hands-on code examples
- Q&A sessions

## 🎨 Customization

### Themes

The presentation uses a custom dark theme. To change themes, edit `index.html`:

```html
<link rel="stylesheet" href="node_modules/reveal.js/dist/theme/black.css" id="theme">
```

Available themes: `black`, `white`, `league`, `beige`, `sky`, `night`, `serif`, `simple`, `solarized`

### Custom Styling

Edit `assets/css/custom.css` to customize:
- Colors and fonts
- Layout and spacing
- Code highlighting
- Animations

### Adding Slides

1. Create a new markdown file in `slides/` directory
2. Add reference in `index.html`:
   ```html
   <section data-markdown="slides/your-slide.md"
            data-separator="^\n---\n$"
            data-separator-vertical="^\n--\n$"
            data-separator-notes="^Note:">
   </section>
   ```

### Slide Separators

In markdown files:
- `---` (three dashes): Horizontal slide separator
- `--` (two dashes): Vertical slide separator
- `Note:`: Speaker notes

## 📊 Features

### Code Highlighting
Syntax highlighting for Python, JavaScript, and other languages using highlight.js.

### Math Support
LaTeX math rendering using MathJax 3:
- Inline: `$equation$`
- Display: `$$equation$$`

### Speaker Notes
Press `S` to open speaker notes view with:
- Current slide
- Next slide preview
- Speaker notes
- Timer

### PDF Export

1. Add `?print-pdf` to the URL: `http://localhost:8000/?print-pdf`
2. Open print dialog (Ctrl/Cmd + P)
3. Set destination to "Save as PDF"
4. Enable "Background graphics"
5. Save

### Slide Overview

Press `Esc` or `O` to see all slides at once for quick navigation.

## 🔧 Development

### Running Locally

```bash
# Install dependencies
npm install

# Start development server
npm start

# The presentation will open at http://localhost:8000
```

### Deploying to GitHub Pages

This presentation is configured for GitHub Pages deployment using CDN-hosted Reveal.js.

#### Automatic Deployment (Recommended)

The repository includes a GitHub Actions workflow that automatically deploys to GitHub Pages on every push to the main branch.

**Setup Steps:**

1. **Enable GitHub Pages**
   - Go to your repository on GitHub
   - Navigate to Settings → Pages
   - Under "Source", select "GitHub Actions"
   - Save the settings

2. **Push your changes**
   ```bash
   git add .
   git commit -m "Configure for GitHub Pages"
   git push origin main
   ```

3. **Wait for deployment**
   - GitHub Actions will automatically build and deploy
   - Check the "Actions" tab to monitor progress
   - Deployment typically takes 1-2 minutes

4. **Access your presentation**
   - Visit: `https://[your-username].github.io/[repo-name]/`
   - Replace with your actual GitHub username and repository name

#### Manual Deployment (Alternative)

If you prefer not to use GitHub Actions:

1. Go to Settings → Pages
2. Under "Source", select "Deploy from a branch"
3. Select branch: `main`, folder: `/ (root)`
4. Save and wait 1-2 minutes
5. Access at: `https://[your-username].github.io/[repo-name]/`

#### Testing Before Deployment

Test the CDN-based version locally:

```bash
# Temporarily rename node_modules to test CDN
mv node_modules node_modules.backup

# Open index.html in your browser
# Test all functionality

# Restore node_modules for development
mv node_modules.backup node_modules
```

#### Deployment Checklist

- [ ] GitHub Pages enabled in repository settings
- [ ] Changes pushed to main branch
- [ ] GitHub Actions workflow completed successfully
- [ ] Presentation loads at GitHub Pages URL
- [ ] All slides navigate correctly
- [ ] Code highlighting works
- [ ] Math equations render
- [ ] Images display correctly
- [ ] No console errors

For detailed deployment instructions, see [GITHUB_PAGES_DEPLOYMENT_PLAN.md](GITHUB_PAGES_DEPLOYMENT_PLAN.md).

### Other Deployment Options

- **Netlify**: Drag and drop the folder or connect your repository
- **Vercel**: Connect your repository for automatic deployments
- **Static hosting**: Any web server (Apache, Nginx, etc.)

## 📚 Resources

### TabPFN
- **Paper**: [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848)
- **GitHub**: [automl/TabPFN](https://github.com/automl/TabPFN)
- **Documentation**: [TabPFN Docs](https://github.com/automl/TabPFN#readme)

### TabICL
- Check the latest research papers and implementations
- Community discussions and improvements

### Reveal.js
- **Documentation**: [revealjs.com](https://revealjs.com/)
- **GitHub**: [hakimel/reveal.js](https://github.com/hakimel/reveal.js)

## 🤝 Contributing

To add or modify content:

1. Edit the relevant markdown file in `slides/`
2. Update `custom.css` for styling changes
3. Test locally with `npm start`
4. Commit your changes

## 📝 License

This presentation is provided as-is for educational and research purposes.

## 🎓 Target Audience

- **Research Audience**: Detailed mathematical formulations and theoretical background
- **Practitioners**: Practical code examples and implementation strategies
- **Students**: Progressive learning from basics to advanced topics

## ⚙️ Technical Requirements

- Modern browser with JavaScript enabled
- Screen resolution: 1920x1080 or higher recommended
- Internet connection (for MathJax CDN)

## 🐛 Troubleshooting

### Slides not loading
- Ensure `npm install` completed successfully
- Check that `node_modules/reveal.js/` exists
- Try clearing browser cache

### Math not rendering
- Check internet connection (MathJax loads from CDN)
- Verify MathJax configuration in `index.html`

### Code highlighting not working
- Ensure highlight.js plugin is loaded
- Check code block language specification

### Speaker notes not showing
- Press `S` to open speaker notes window
- Allow pop-ups in your browser
- Check browser console for errors

## 📧 Contact

For questions or feedback about this presentation, please open an issue or contact the maintainer.

---

**Happy Presenting! 🎉**
