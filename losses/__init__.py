from torch import nn
from importlib import import_module


class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.loss = []
        self.loss_module = nn.ModuleList()

        for loss in args.loss.split('+'):
            loss_function = None
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'Charb':
                module = import_module('losses.charbonnier')
                loss_function = getattr(module, 'Charbonnier')()
            elif loss_type == 'Lap':
                module = import_module('losses.laplacianpyramid')
                loss_function = getattr(module, 'LaplacianLoss')()
            elif loss_type.find('GAN') >= 0:
                module = import_module('losses.adversarial')
                loss_function = getattr(module, 'Adversarial')(args, loss_type)

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.loss_module.to('cuda')

    def forward(self, output, gt, input_frames):
        losses = []
        for l in self.loss:
            if l['function'] is not None:
                if l['type'] in ['FI_GAN', 'FI_Cond_GAN', 'STGAN']:
                    loss = l['function'](output['frame1'], gt, input_frames)
                else:
                    loss = l['function'](output['frame1'], gt)

                effective_loss = l['weight'] * loss
                losses.append(effective_loss)

        loss_sum = sum(losses)

        return loss_sum


class DistillationLoss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(DistillationLoss, self).__init__()

        self.loss = []
        self.loss_module = nn.ModuleList()
        self.temp = args.temp
        self.alpha = args.alpha

        for loss in args.distill_loss_fn.split('+'):
            distill_loss_function = None
            distill_weight, distill_loss_type = loss.split('*')
            if distill_loss_type == 'MSE':
                distill_loss_function = nn.MSELoss()
            elif distill_loss_type == 'KLDivergence':
                distill_loss_function = nn.KLDivLoss(reduction="mean")
            elif distill_loss_type == 'CrossEntropy':
                distill_loss_function = nn.CrossEntropyLoss()
            elif distill_loss_type == 'L1':
                distill_loss_function = nn.L1Loss()
            elif distill_loss_type == 'Charb':
                module = import_module('losses.charbonnier')
                distill_loss_function = getattr(module, 'Charbonnier')()
            elif distill_loss_type == 'Lap':
                module = import_module('losses.laplacianpyramid')
                distill_loss_function = getattr(module, 'LaplacianLoss')()
            elif distill_loss_type.find('GAN') >= 0:
                module = import_module('losses.adversarial')
                distill_loss_function = getattr(module, 'Adversarial')(args, distill_loss_type)

        for loss in args.student_loss_fn.split('+'):
            student_loss_function = None
            student_weight, student_loss_type = loss.split('*')
            if student_loss_type == 'MSE':
                student_loss_function = nn.MSELoss()
            elif student_loss_type == 'KLDivergence':
                student_loss_function = nn.KLDivLoss(reduction="mean")
            elif student_loss_type == 'CrossEntropy':
                student_loss_function = nn.CrossEntropyLoss()
            elif student_loss_type == 'L1':
                student_loss_function = nn.L1Loss()
            elif student_loss_type == 'Charb':
                module = import_module('losses.charbonnier')
                student_loss_function = getattr(module, 'Charbonnier')()
            elif student_loss_type == 'Lap':
                module = import_module('losses.laplacianpyramid')
                student_loss_function = getattr(module, 'LaplacianLoss')()
            elif student_loss_type.find('GAN') >= 0:
                module = import_module('losses.adversarial')
                student_loss_function = getattr(module, 'Adversarial')(args, student_loss_type)

            self.loss.append({
                'distill_type': distill_loss_type,
                'distill_weight': float(distill_weight),
                'distill_function': distill_loss_function,
                'student_type': student_loss_type,
                'student_weight': float(student_weight),
                'student_function': student_loss_function}
            )

        for l in self.loss:
            if l['distill_function'] is not None:
                print('{:.3f} * {}'.format(l['student_weight'], l['student_type']))
                self.loss_module.append(l['distill_function'])
            if l['student_function'] is not None:
                print('{:.3f} * {}'.format(l['distill_weight'], l['distill_type']))
                self.loss_module.append(l['student_function'])

        self.loss_module.to('cuda')

    def forward(self, student_output, teacher_output, gt, input_frames):

        # distillation loss function
        def distillation_loss(student_pred, teacher_pred, distill_loss_fn, temperature=10):
            soft_pred = softmax(student_pred / temperature)
            soft_labels = softmax(teacher_pred / temperature)
            loss = distill_loss_fn(soft_pred, soft_labels) * temperature**2
            return loss

        softmax = nn.Softmax(dim=0)

        losses = []
        for l in self.loss:

            student_loss = l['student_function'](
                gt,
                softmax(student_output['frame1'])
            )
            # print(student_loss)
            distill_loss = distillation_loss(
                student_pred = student_output['frame1'],
                teacher_pred = teacher_output['frame1'],
                distill_loss_fn = l['distill_function'],
                temperature=self.temp
            )
            # print(distill_loss)
            effective_loss = self.alpha * student_loss + \
                (1 - self.alpha) * distill_loss

            losses.append(effective_loss)

        loss_sum = sum(losses)

        return loss_sum
