from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base() # sets as the shared page for both tables and ensures that it's registered under the same metadata