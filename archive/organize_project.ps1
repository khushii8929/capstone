# ============================================================================
# PROJECT ORGANIZATION SCRIPT - Ahmedabad Real Estate Analytics
# ============================================================================
# This script organizes the project into a modular folder structure
# ============================================================================

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "*" * 78 -ForegroundColor Cyan
Write-Host "  ORGANIZING PROJECT INTO MODULAR STRUCTURE" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "*" * 78 -ForegroundColor Cyan
Write-Host ""

# Function to move files safely
function Move-FilesSafely {
    param(
        [string]$Source,
        [string]$Destination,
        [string]$Description
    )
    
    Write-Host "📁 Moving $Description..." -ForegroundColor Cyan
    
    if (Test-Path $Source) {
        Move-Item -Path $Source -Destination $Destination -Force
        Write-Host "   ✅ Moved: $(Split-Path $Source -Leaf)" -ForegroundColor Green
    }
}

# ============================================================================
# 1. ORGANIZE DATA FILES
# ============================================================================
Write-Host "`n📊 1. Organizing Data Files..." -ForegroundColor Yellow

# Raw data
if (Test-Path "ahmedabad_real_estate_data.csv") {
    Move-Item "ahmedabad_real_estate_data.csv" "data/raw/" -Force
    Write-Host "   ✅ Moved: ahmedabad_real_estate_data.csv → data/raw/" -ForegroundColor Green
}

# Processed data
$processedData = @(
    "cleaned_real_estate_data.csv",
    "featured_real_estate_data.csv",
    "final_analysis_with_predictions.csv",
    "model_comparison_results.csv",
    "feature_importance.csv"
)

foreach ($file in $processedData) {
    if (Test-Path $file) {
        Move-Item $file "data/processed/" -Force
        Write-Host "   ✅ Moved: $file → data/processed/" -ForegroundColor Green
    }
}

# ============================================================================
# 2. ORGANIZE NOTEBOOKS
# ============================================================================
Write-Host "`n📓 2. Organizing Notebooks..." -ForegroundColor Yellow

$notebooks = @(
    "00_MASTER_PIPELINE.ipynb",
    "01_data_cleaning.ipynb",
    "02_feature_engineering.ipynb",
    "03_exploratory_data_analysis.ipynb",
    "04_machine_learning_models.ipynb",
    "05_business_insights_usecases.ipynb",
    "06_model_visualizations_summary.ipynb"
)

foreach ($notebook in $notebooks) {
    if (Test-Path $notebook) {
        Move-Item $notebook "notebooks/" -Force
        Write-Host "   ✅ Moved: $notebook → notebooks/" -ForegroundColor Green
    }
}

# ============================================================================
# 3. ORGANIZE MODEL FILES
# ============================================================================
Write-Host "`n🤖 3. Organizing Model Files..." -ForegroundColor Yellow

$modelFiles = Get-ChildItem -Filter "*.pkl"
foreach ($file in $modelFiles) {
    Move-Item $file.FullName "models/" -Force
    Write-Host "   ✅ Moved: $($file.Name) → models/" -ForegroundColor Green
}

# ============================================================================
# 4. ORGANIZE VISUALIZATIONS
# ============================================================================
Write-Host "`n📈 4. Organizing Visualizations..." -ForegroundColor Yellow

# EDA visualizations
$edaViz = @(
    "01_price_distribution.png",
    "02_area_vs_price.png",
    "03_top_localities.png",
    "04_bhk_distribution.png",
    "05_furnishing_impact.png",
    "07_correlation_heatmap.png"
)

foreach ($viz in $edaViz) {
    if (Test-Path $viz) {
        Move-Item $viz "visualizations/eda/" -Force
        Write-Host "   ✅ Moved: $viz → visualizations/eda/" -ForegroundColor Green
    }
}

# Model performance visualizations
$modelViz = @(
    "06_model_comparison.png",
    "06_price_per_sqft_localities.png",
    "07_actual_vs_predicted.png",
    "08_residual_plot.png",
    "09_residual_distribution.png",
    "10_error_percentage.png",
    "11_feature_importance.png",
    "12_error_by_price_range.png"
)

foreach ($viz in $modelViz) {
    if (Test-Path $viz) {
        Move-Item $viz "visualizations/model_performance/" -Force
        Write-Host "   ✅ Moved: $viz → visualizations/model_performance/" -ForegroundColor Green
    }
}

# Master dashboard visualizations
$masterViz = @(
    "00_master_market_overview.png",
    "00_master_segment_analysis.png",
    "00_master_location_intelligence.png",
    "00_master_model_dashboard.png",
    "00_master_investment_opportunities.png",
    "00_master_train_vs_test_accuracy.png"
)

foreach ($viz in $masterViz) {
    if (Test-Path $viz) {
        Move-Item $viz "visualizations/master_dashboard/" -Force
        Write-Host "   ✅ Moved: $viz → visualizations/master_dashboard/" -ForegroundColor Green
    }
}

# ============================================================================
# 5. ORGANIZE SCRIPTS
# ============================================================================
Write-Host "`n🔧 5. Organizing Scripts..." -ForegroundColor Yellow

$scripts = @(
    "scraper.py",
    "run_complete_pipeline.py",
    "create_analysis_report.py",
    "generate_detailed_analysis.py"
)

foreach ($script in $scripts) {
    if (Test-Path $script) {
        Move-Item $script "scripts/" -Force
        Write-Host "   ✅ Moved: $script → scripts/" -ForegroundColor Green
    }
}

# ============================================================================
# 6. ORGANIZE DOCUMENTATION
# ============================================================================
Write-Host "`n📄 6. Organizing Documentation..." -ForegroundColor Yellow

$docs = @(
    "README.md",
    "PROJECT_SUMMARY.md",
    "VISUALIZATION_CATALOG.md"
)

foreach ($doc in $docs) {
    if (Test-Path $doc) {
        Move-Item $doc "docs/" -Force
        Write-Host "   ✅ Moved: $doc → docs/" -ForegroundColor Green
    }
}

# ============================================================================
# 7. ORGANIZE REPORTS
# ============================================================================
Write-Host "`n📊 7. Organizing Reports..." -ForegroundColor Yellow

$reports = @(
    "FINAL_PROJECT_REPORT.md",
    "COMPREHENSIVE_EDA_ANALYSIS.txt"
)

foreach ($report in $reports) {
    if (Test-Path $report) {
        Move-Item $report "reports/" -Force
        Write-Host "   ✅ Moved: $report → reports/" -ForegroundColor Green
    }
}

# ============================================================================
# 8. CREATE INDEX FILES
# ============================================================================
Write-Host "`n📋 8. Creating Index Files..." -ForegroundColor Yellow

# Create README for each directory
$readmeContent = @"
# Directory Contents

This directory is part of the Ahmedabad Real Estate Analytics project.

For main documentation, see: docs/README.md
"@

$directories = @("data", "notebooks", "models", "visualizations", "scripts", "src", "reports")
foreach ($dir in $directories) {
    if (!(Test-Path "$dir/README.md")) {
        $readmeContent | Out-File "$dir/README.md" -Encoding UTF8
        Write-Host "   ✅ Created: $dir/README.md" -ForegroundColor Green
    }
}

# ============================================================================
# 9. CREATE PROJECT STRUCTURE DOCUMENT
# ============================================================================
Write-Host "`n📁 9. Creating Project Structure Documentation..." -ForegroundColor Yellow

$structureDoc = @'
# Project Structure - Ahmedabad Real Estate Analytics

## 📂 Directory Organization

\`\`\`
Caapstone-Phase1/
│
├── data/                           # All data files
│   ├── raw/                        # Original scraped data
│   │   └── ahmedabad_real_estate_data.csv
│   └── processed/                  # Cleaned and feature-engineered data
│       ├── cleaned_real_estate_data.csv
│       ├── featured_real_estate_data.csv
│       ├── final_analysis_with_predictions.csv
│       ├── model_comparison_results.csv
│       └── feature_importance.csv
│
├── notebooks/                      # Jupyter notebooks (analysis workflow)
│   ├── 00_MASTER_PIPELINE.ipynb   # Master orchestration notebook
│   ├── 01_data_cleaning.ipynb     # Data cleaning & preprocessing
│   ├── 02_feature_engineering.ipynb # Feature creation
│   ├── 03_exploratory_data_analysis.ipynb # EDA visualizations
│   ├── 04_machine_learning_models.ipynb # Model training & comparison
│   ├── 05_business_insights_usecases.ipynb # Business insights
│   └── 06_model_visualizations_summary.ipynb # Model visualizations
│
├── models/                         # Trained ML models and artifacts
│   ├── best_model_GradientBoosting.pkl # Best trained model
│   ├── feature_scaler.pkl         # Feature scaling transformer
│   ├── feature_columns.pkl        # Feature list
│   └── model_info.pkl             # Model metadata
│
├── visualizations/                 # All generated visualizations
│   ├── eda/                       # Exploratory data analysis charts
│   │   ├── 01_price_distribution.png
│   │   ├── 02_area_vs_price.png
│   │   ├── 03_top_localities.png
│   │   ├── 04_bhk_distribution.png
│   │   ├── 05_furnishing_impact.png
│   │   └── 07_correlation_heatmap.png
│   │
│   ├── model_performance/         # Model performance charts
│   │   ├── 06_model_comparison.png
│   │   ├── 07_actual_vs_predicted.png
│   │   ├── 08_residual_plot.png
│   │   ├── 09_residual_distribution.png
│   │   ├── 10_error_percentage.png
│   │   ├── 11_feature_importance.png
│   │   └── 12_error_by_price_range.png
│   │
│   └── master_dashboard/          # Executive dashboard visualizations
│       ├── 00_master_market_overview.png
│       ├── 00_master_segment_analysis.png
│       ├── 00_master_location_intelligence.png
│       ├── 00_master_model_dashboard.png
│       ├── 00_master_investment_opportunities.png
│       └── 00_master_train_vs_test_accuracy.png
│
├── scripts/                        # Utility scripts
│   ├── scraper.py                 # Web scraping script
│   ├── run_complete_pipeline.py   # Pipeline automation
│   ├── create_analysis_report.py  # Report generation
│   └── generate_detailed_analysis.py # Detailed analysis
│
├── src/                           # Source code modules (future expansion)
│   └── README.md
│
├── docs/                          # Project documentation
│   ├── README.md                  # Main project documentation
│   ├── PROJECT_SUMMARY.md         # Project overview
│   └── VISUALIZATION_CATALOG.md   # Visualization index
│
├── reports/                       # Analysis reports
│   ├── FINAL_PROJECT_REPORT.md    # Final comprehensive report
│   └── COMPREHENSIVE_EDA_ANALYSIS.txt # EDA findings
│
└── organize_project.ps1           # This organization script

\`\`\`

## 🔄 Workflow

1. **Data Collection**: \`scripts/scraper.py\` → \`data/raw/\`
2. **Data Cleaning**: \`notebooks/01_data_cleaning.ipynb\` → \`data/processed/cleaned_*.csv\`
3. **Feature Engineering**: \`notebooks/02_feature_engineering.ipynb\` → \`data/processed/featured_*.csv\`
4. **EDA**: \`notebooks/03_exploratory_data_analysis.ipynb\` → \`visualizations/eda/\`
5. **Model Training**: \`notebooks/04_machine_learning_models.ipynb\` → \`models/\`
6. **Business Insights**: \`notebooks/05_business_insights_usecases.ipynb\`
7. **Master Dashboard**: \`notebooks/00_MASTER_PIPELINE.ipynb\` → \`visualizations/master_dashboard/\`

## 📊 Key Deliverables

- **Best Model**: 99.29% accuracy (R² Score)
- **Dataset**: 2,247 properties across Ahmedabad
- **Visualizations**: 20+ comprehensive charts
- **Business Insights**: 6 actionable use cases

## 🚀 Quick Start

1. Open \`notebooks/00_MASTER_PIPELINE.ipynb\` for complete workflow
2. Check \`docs/README.md\` for detailed documentation
3. View \`reports/FINAL_PROJECT_REPORT.md\` for findings

---

**Project Status**: ✅ Production Ready
**Last Updated**: November 2025
'@

$structureDoc | Out-File "PROJECT_STRUCTURE.md" -Encoding UTF8
Write-Host "   ✅ Created: PROJECT_STRUCTURE.md" -ForegroundColor Green

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
Write-Host "  ✅ PROJECT ORGANIZATION COMPLETE!" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Cyan

Write-Host "`n📁 New Project Structure:" -ForegroundColor Yellow
Write-Host "   ├── data/              (Raw & Processed datasets)" -ForegroundColor White
Write-Host "   ├── notebooks/         (Jupyter analysis notebooks)" -ForegroundColor White
Write-Host "   ├── models/            (Trained ML models)" -ForegroundColor White
Write-Host "   ├── visualizations/    (All charts & graphs)" -ForegroundColor White
Write-Host "   ├── scripts/           (Utility scripts)" -ForegroundColor White
Write-Host "   ├── src/               (Source code modules)" -ForegroundColor White
Write-Host "   ├── docs/              (Documentation)" -ForegroundColor White
Write-Host "   └── reports/           (Analysis reports)" -ForegroundColor White

Write-Host "`n📖 Next Steps:" -ForegroundColor Yellow
Write-Host "   1. Review PROJECT_STRUCTURE.md for complete organization" -ForegroundColor White
Write-Host "   2. Check each directory's README.md for specific contents" -ForegroundColor White
Write-Host "   3. Open notebooks/00_MASTER_PIPELINE.ipynb to start analysis" -ForegroundColor White

Write-Host "`n🎉 Your project is now professionally organized!" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host ""
