# Coding Instructions

## Code Quality
- Clean, readable, production-ready code
- Meaningful names, small functions
- Type hints/annotations
- Error handling by default

## Security
- Environment variables for secrets
- Validate/sanitize inputs
- Parameterized SQL queries
- No hardcoded credentials

## Best Practices
- DRY: Don't repeat yourself
- Single responsibility per function
- Use appropriate data structures
- Close resources (files, connections)
- Log errors, don't print

## Structure
- Separate business logic from I/O
- Use dependency injection
- Add docstrings for public functions
- Constants for magic values

## Language Specific

**Python:**
- Virtual environments
- Type hints
- Context managers
- List comprehensions (when clear)

**JavaScript/TypeScript:**
- async/await over callbacks
- const by default
- Destructuring
- Handle promise rejections

## API Development
- Proper HTTP status codes
- Request validation
- Rate limiting
- Versioning

## Database
- Connection pooling
- Indexes on queried fields
- Transactions for multi-step ops
- Avoid SELECT *

## What to Avoid
- Deprecated libraries
- Ignoring error cases
- Mixed concerns
- Skipping validation
- Exposing secrets in logs

## Response Format
- Complete, runnable code
- Installation commands included
- Comments only for complex logic