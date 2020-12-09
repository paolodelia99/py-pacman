from .controller import Controller


def main(args):
    controller = Controller(layout_name=args.layout[0], act_sound=args.sound, act_state=args.state, ai_agent=None)
    controller.load_menu()
