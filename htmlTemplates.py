css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
.name {
  text-align: center;
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://media.discordapp.net/attachments/1040216906231197796/1115742410467708958/Dadle_a_chatbot_icon_for_the_AI_jarvis_approchable_friendly_and_21fdb40f-8bde-45de-8a44-1afba21dd542.png?width=910&height=910">
        <p class="name">Jarvis</p>
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://media.discordapp.net/attachments/1040216906231197796/1115746519052722267/Dadle_A_user_icon_for_chat_representing_a_business_consultant_w_7eb8e4a0-103c-4d68-baf5-b5388cf539db.png?width=910&height=910">
        <p User class="name">User</p>
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''