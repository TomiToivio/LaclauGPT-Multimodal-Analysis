PRIVATE=/scratch/project_2009497/LaclauGPT-Private

# Find files that define MongoDB connection settings
grep -rlE 'LACLAUGPT_MONGO_URI|LACLAUGPT_MONGODB_URI|MONGODB_URI' \
  "$PRIVATE" \
  --include='*.env' \
  --include='*.sh' \
  --include='*.yaml' \
  --include='*.toml' \
  --exclude='*.example' \
  2>/dev/null
