from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from shield_rca.agents import supabase

security = HTTPBearer()

async def verify_supabase_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verifies the JWT token using Supabase and returns the user object."""
    token = credentials.credentials
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")
        
    # Verify the token by calling Supabase Auth API
    # get_user() automatically verifies the JWT against the Supabase project
    response = supabase.auth.get_user(token)
    
    if not response.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    return response.user
