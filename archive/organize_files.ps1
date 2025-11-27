# ============================================================================
# PROJECT ORGANIZATION SCRIPT - Ahmedabad Real Estate Analytics
# ============================================================================

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  ORGANIZING PROJECT INTO MODULAR STRUCTURE" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# 1. ORGANIZE DATA FILES
# ============================================================================
Write-Host "`n1. Organizing Data Files..." -ForegroundColor Yellow

# Raw data
if (Test-Path "ahmedabad_real_estate_data.csv") {
    Move-Item "ahmedabad_real_estate_data.csv" "data/raw/" -Force
    Write-Host "   OK Moved: ahmedabad_real_estate_data.csv -> data/raw/" -ForegroundColor Green
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
        Write-Host "   OK Moved: $file -> data/processed/" -ForegroundColor Green
    }
}

# ============================================================================
# 2. ORGANIZE NOTEBOOKS
# ============================================================================
Write-Host "`n2. Organizing Notebooks..." -ForegroundColor Yellow

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
        Write-Host "   OK Moved: $notebook -> notebooks/" -ForegroundColor Green
    }
}

# ============================================================================
# 3. ORGANIZE MODEL FILES
# ============================================================================
Write-Host "`n3. Organizing Model Files..." -ForegroundColor Yellow

$modelFiles = Get-ChildItem -Filter "*.pkl" -ErrorAction SilentlyContinue
foreach ($file in $modelFiles) {
    Move-Item $file.FullName "models/" -Force
    Write-Host "   OK Moved: $($file.Name) -> models/" -ForegroundColor Green
}

# ============================================================================
# 4. ORGANIZE VISUALIZATIONS
# ============================================================================
Write-Host "`n4. Organizing Visualizations..." -ForegroundColor Yellow

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
        Write-Host "   OK Moved: $viz -> visualizations/eda/" -ForegroundColor Green
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
        Write-Host "   OK Moved: $viz -> visualizations/model_performance/" -ForegroundColor Green
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
        Write-Host "   OK Moved: $viz -> visualizations/master_dashboard/" -ForegroundColor Green
    }
}

# ============================================================================
# 5. ORGANIZE SCRIPTS
# ============================================================================
Write-Host "`n5. Organizing Scripts..." -ForegroundColor Yellow

$scripts = @(
    "scraper.py",
    "run_complete_pipeline.py",
    "create_analysis_report.py",
    "generate_detailed_analysis.py"
)

foreach ($script in $scripts) {
    if (Test-Path $script) {
        Move-Item $script "scripts/" -Force
        Write-Host "   OK Moved: $script -> scripts/" -ForegroundColor Green
    }
}

# ============================================================================
# 6. ORGANIZE DOCUMENTATION
# ============================================================================
Write-Host "`n6. Organizing Documentation..." -ForegroundColor Yellow

$docs = @(
    "README.md",
    "PROJECT_SUMMARY.md",
    "VISUALIZATION_CATALOG.md"
)

foreach ($doc in $docs) {
    if (Test-Path $doc) {
        Move-Item $doc "docs/" -Force
        Write-Host "   OK Moved: $doc -> docs/" -ForegroundColor Green
    }
}

# ============================================================================
# 7. ORGANIZE REPORTS
# ============================================================================
Write-Host "`n7. Organizing Reports..." -ForegroundColor Yellow

$reports = @(
    "FINAL_PROJECT_REPORT.md",
    "COMPREHENSIVE_EDA_ANALYSIS.txt"
)

foreach ($report in $reports) {
    if (Test-Path $report) {
        Move-Item $report "reports/" -Force
        Write-Host "   OK Moved: $report -> reports/" -ForegroundColor Green
    }
}

# ============================================================================
# FINAL SUMMARY
# ============================================================================
Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  PROJECT ORGANIZATION COMPLETE!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan

Write-Host "`nNew Project Structure:" -ForegroundColor Yellow
Write-Host "   data/              (Raw & Processed datasets)" -ForegroundColor White
Write-Host "   notebooks/         (Jupyter analysis notebooks)" -ForegroundColor White
Write-Host "   models/            (Trained ML models)" -ForegroundColor White
Write-Host "   visualizations/    (All charts & graphs)" -ForegroundColor White
Write-Host "   scripts/           (Utility scripts)" -ForegroundColor White
Write-Host "   src/               (Source code modules)" -ForegroundColor White
Write-Host "   docs/              (Documentation)" -ForegroundColor White
Write-Host "   reports/           (Analysis reports)" -ForegroundColor White

Write-Host "`nNext Steps:" -ForegroundColor Yellow
Write-Host "   1. Check MAIN_README.md for complete project overview" -ForegroundColor White
Write-Host "   2. Open notebooks/00_MASTER_PIPELINE.ipynb to start analysis" -ForegroundColor White
Write-Host "   3. Review docs/ folder for detailed documentation" -ForegroundColor White

Write-Host "`nYour project is now professionally organized!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
