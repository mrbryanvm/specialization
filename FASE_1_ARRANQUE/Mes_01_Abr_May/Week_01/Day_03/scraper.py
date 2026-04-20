import requests

class UserScraper:
    """
    A class to fetch and display user data from a remote API.
    """
    def __init__(self):
        # Base URL for the users endpoint
        self.base_url = "https://jsonplaceholder.typicode.com/users"

    def fetch_users(self):
        """
        Connects to the API and returns a list of users.
        """
        try:
            response = requests.get(self.base_url)
            if response.status_code == 200:
                # Returns the full list of user dictionaries
                return response.json()
            else:
                print(f"Failed to fetch data. Status: {response.status_code}")
                return []
        except Exception as e:
            print(f"An error occurred during fetch: {e}")
            return []

    def display_emails(self, users_list):
        """
        Iterates through the user list and prints name and email.
        """
        print("\n--- User Directory ---")
        for user in users_list:
            # We access the dictionary keys 'name' and 'email'
            print(f"- {user['name']} ({user['email']})")
        print("----------------------\n")

if __name__ == "__main__":
    # 1. Instantiate the class
    scraper = UserScraper()
    
    # 2. Get the user data
    users = scraper.fetch_users()
    
    # 3. Display the formatted results
    if users:
        scraper.display_emails(users)