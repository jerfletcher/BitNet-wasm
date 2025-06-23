#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
VERSION_TYPE="patch"
PUSH=false
CURRENT_VERSION=""

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE    Version increment type: major, minor, patch (default: patch)"
    echo "  -p, --push         Push the tag and changes to remote repository"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                     # Increment patch version (1.0.0 -> 1.0.1)"
    echo "  $0 -t minor            # Increment minor version (1.0.0 -> 1.1.0)"
    echo "  $0 -t major -p         # Increment major version and push to remote"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            VERSION_TYPE="$2"
            if [[ ! "$VERSION_TYPE" =~ ^(major|minor|patch)$ ]]; then
                echo -e "${RED}Error: Invalid version type '$VERSION_TYPE'. Must be major, minor, or patch.${NC}"
                exit 1
            fi
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option '$1'${NC}"
            usage
            exit 1
            ;;
    esac
done

# Function to get current version from package.json
get_current_version() {
    if [[ ! -f "package.json" ]]; then
        echo -e "${RED}Error: package.json not found${NC}"
        exit 1
    fi
    
    CURRENT_VERSION=$(grep '"version"' package.json | sed 's/.*"version": *"\([^"]*\)".*/\1/')
    if [[ -z "$CURRENT_VERSION" ]]; then
        echo -e "${RED}Error: Could not find version in package.json${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}Current version: $CURRENT_VERSION${NC}"
}

# Function to increment version
increment_version() {
    local version=$1
    local type=$2
    
    # Split version into parts
    IFS='.' read -ra VERSION_PARTS <<< "$version"
    local major=${VERSION_PARTS[0]}
    local minor=${VERSION_PARTS[1]}
    local patch=${VERSION_PARTS[2]}
    
    case $type in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
    esac
    
    echo "$major.$minor.$patch"
}

# Function to update version in package.json
update_package_json() {
    local new_version=$1
    echo -e "${YELLOW}Updating package.json...${NC}"
    
    # Create backup
    cp package.json package.json.bak
    
    # Update version using sed
    sed -i.tmp "s/\"version\": *\"[^\"]*\"/\"version\": \"$new_version\"/" package.json
    rm package.json.tmp
    
    echo -e "${GREEN}✓ Updated package.json${NC}"
}

# Function to update version in GitHub workflow
update_github_workflow() {
    local new_version=$1
    local workflow_file=".github/workflows/build-and-publish.yml"
    
    if [[ -f "$workflow_file" ]]; then
        echo -e "${YELLOW}Updating GitHub workflow...${NC}"
        
        # Create backup
        cp "$workflow_file" "$workflow_file.bak"
        
        # Update version in the workflow file
        sed -i.tmp "s/\"version\": *\"[^\"]*\"/\"version\": \"$new_version\"/" "$workflow_file"
        rm "$workflow_file.tmp"
        
        echo -e "${GREEN}✓ Updated GitHub workflow${NC}"
    fi
}

# Function to update version in any documentation files
update_documentation() {
    local new_version=$1
    
    # Update README.md if it contains version references
    if [[ -f "README.md" ]] && grep -q "version.*[0-9]\+\.[0-9]\+\.[0-9]\+" README.md; then
        echo -e "${YELLOW}Updating README.md...${NC}"
        cp README.md README.md.bak
        
        # Update version badges or version references
        sed -i.tmp "s/version-[0-9]\+\.[0-9]\+\.[0-9]\+/version-$new_version/g" README.md
        sed -i.tmp "s/v[0-9]\+\.[0-9]\+\.[0-9]\+/v$new_version/g" README.md
        rm README.md.tmp
        
        echo -e "${GREEN}✓ Updated README.md${NC}"
    fi
}

# Function to create git tag and commit
create_tag_and_commit() {
    local new_version=$1
    local tag_name="v$new_version"
    
    echo -e "${YELLOW}Creating git commit and tag...${NC}"
    
    # Check if there are any changes to commit
    if ! git diff --quiet || ! git diff --cached --quiet; then
        # Stage the modified files
        git add package.json
        [[ -f ".github/workflows/build-and-publish.yml" ]] && git add .github/workflows/build-and-publish.yml
        [[ -f "README.md" ]] && git add README.md
        
        # Commit the changes
        git commit -m "chore: bump version to $new_version"
        echo -e "${GREEN}✓ Created commit for version $new_version${NC}"
    fi
    
    # Create the tag
    git tag -a "$tag_name" -m "Release $tag_name"
    echo -e "${GREEN}✓ Created tag $tag_name${NC}"
    
    if [[ "$PUSH" == true ]]; then
        echo -e "${YELLOW}Pushing to remote repository...${NC}"
        
        # Push the commit and tag
        git push origin $(git branch --show-current)
        git push origin "$tag_name"
        
        echo -e "${GREEN}✓ Pushed commit and tag to remote${NC}"
        echo -e "${BLUE}Release $tag_name has been published!${NC}"
    else
        echo -e "${YELLOW}To push the release, run:${NC}"
        echo -e "${BLUE}  git push origin $(git branch --show-current)${NC}"
        echo -e "${BLUE}  git push origin $tag_name${NC}"
    fi
}

# Main execution
main() {
    echo -e "${BLUE}🚀 BitNet-WASM Release Tool${NC}"
    echo ""
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        echo -e "${RED}Error: Not in a git repository${NC}"
        exit 1
    fi
    
    # Check for uncommitted changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo -e "${YELLOW}Warning: You have uncommitted changes.${NC}"
        echo -e "${YELLOW}These will be included in the release commit.${NC}"
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Aborted.${NC}"
            exit 0
        fi
    fi
    
    # Get current version
    get_current_version
    
    # Calculate new version
    NEW_VERSION=$(increment_version "$CURRENT_VERSION" "$VERSION_TYPE")
    echo -e "${GREEN}New version: $NEW_VERSION${NC}"
    echo ""
    
    # Confirm before proceeding
    echo -e "${YELLOW}This will:${NC}"
    echo "  • Update version in package.json"
    echo "  • Update version in GitHub workflow"
    echo "  • Update version references in documentation"
    echo "  • Create a git commit with these changes"
    echo "  • Create a git tag v$NEW_VERSION"
    if [[ "$PUSH" == true ]]; then
        echo "  • Push the commit and tag to remote repository"
    fi
    echo ""
    
    read -p "Proceed with $VERSION_TYPE version bump to $NEW_VERSION? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Aborted.${NC}"
        exit 0
    fi
    
    # Update version in all files
    update_package_json "$NEW_VERSION"
    update_github_workflow "$NEW_VERSION"
    update_documentation "$NEW_VERSION"
    
    # Create tag and commit
    create_tag_and_commit "$NEW_VERSION"
    
    echo ""
    echo -e "${GREEN}🎉 Successfully created release $NEW_VERSION!${NC}"
}

# Run main function
main "$@"
