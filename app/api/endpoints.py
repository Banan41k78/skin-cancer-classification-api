# Добавьте эти эндпоинты в router

@router.get("/sex-options")
async def get_sex_options():
    """Get available sex options"""
    return model_manager.get_sex_options()

@router.get("/localization-options")
async def get_localization_options():
    """Get available localization options"""
    return model_manager.get_localization_options()

@router.get("/dx-type-options")
async def get_dx_type_options():
    """Get available diagnosis type options"""
    return model_manager.get_dx_type_options()

@router.get("/diagnosis-classes")
async def get_diagnosis_classes():
    """Get diagnosis classes information"""
    return {
        "diagnosis_classes": model_manager.get_diagnosis_classes(),
        "total_classes": len(model_manager.get_diagnosis_classes())
    }

@router.get("/risk-classes")
async def get_risk_classes():
    """Get risk classes information"""
    return {
        "risk_classes": model_manager.get_risk_classes(),
        "total_classes": len(model_manager.get_risk_classes())
    }